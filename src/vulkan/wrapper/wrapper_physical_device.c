#include <math.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>

#include "wrapper_private.h"
#include "wrapper_log.h"
#include "wrapper_entrypoints.h"
#include "wrapper_trampolines.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_dispatch_table.h"
#include "vk_extensions.h"
#include "vk_physical_device.h"
#include "vk_util.h"
#include "wsi_common.h"
#include "util/os_misc.h"

static uint32_t
parse_vk_version_from_env()
{
   uint32_t apiVersion = 0, major = 0, minor = 0, patch = 0;

   const char *wrapper_vk_version = getenv("WRAPPER_VK_VERSION");

   if (wrapper_vk_version) {
      sscanf(wrapper_vk_version, "%d.%d.%d", &major, &minor, &patch);
      apiVersion = VK_MAKE_VERSION(major, minor, patch);
   }

   return apiVersion;
}

/* The Mali driver's subgroup operations are hidden by default (historical
 * wrapper behaviour); WRAPPER_SUBGROUP_OPS=1 passes the real ones through. */
static bool
wrapper_subgroup_ops_passthrough(void)
{
   static int passthrough = -1;

   if (passthrough == -1)
      passthrough = getenv("WRAPPER_SUBGROUP_OPS") && atoi(getenv("WRAPPER_SUBGROUP_OPS"));

   return passthrough != 0;
}

static void
format_driver_version(const uint32_t driverVersion, char *out, size_t out_size)
{
   snprintf(out, out_size, "%u.%u.%u",
            VK_API_VERSION_MAJOR(driverVersion),
            VK_API_VERSION_MINOR(driverVersion),
            VK_API_VERSION_PATCH(driverVersion));
}

/* The Vulkan driver this wrapper sits on must be Arm's Mali driver. */
static bool
wrapper_is_mali_device(const VkPhysicalDeviceProperties *props,
                       const VkPhysicalDeviceDriverProperties *driver)
{
   return props->vendorID == WRAPPER_MALI_VENDOR_ID ||
          driver->driverID == VK_DRIVER_ID_ARM_PROPRIETARY;
}

static VkResult
wrapper_setup_device_extensions(struct wrapper_physical_device *pdevice) {
   struct vk_device_extension_table *exts = &pdevice->vk.supported_extensions;
   uint32_t pdevice_extension_count = 0;
   VkExtensionProperties *pdevice_extensions;
   VkResult result;

   result = pdevice->dispatch_table.EnumerateDeviceExtensionProperties(
      pdevice->dispatch_handle, NULL, &pdevice_extension_count, NULL);
   if (result != VK_SUCCESS)
      return result;

   pdevice_extensions = malloc((pdevice_extension_count + 1) * sizeof(*pdevice_extensions));
   if (!pdevice_extensions)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   result = pdevice->dispatch_table.EnumerateDeviceExtensionProperties(
      pdevice->dispatch_handle, NULL, &pdevice_extension_count, pdevice_extensions);
   if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
      free(pdevice_extensions);
      return result;
   }

   *exts = wrapper_device_extensions;

   for (uint32_t i = 0; i < pdevice_extension_count; i++) {
      int idx;
      for (idx = 0; idx < VK_DEVICE_EXTENSION_COUNT; idx++) {
         if (strcmp(vk_device_extensions[idx].extensionName,
                     pdevice_extensions[i].extensionName) == 0)
            break;
      }

      if (idx >= VK_DEVICE_EXTENSION_COUNT)
         continue;

      if (wrapper_filter_extensions.extensions[idx])
         continue;

      pdevice->base_supported_extensions.extensions[idx] =
         exts->extensions[idx] = true;
   }

   free(pdevice_extensions);

   exts->KHR_present_wait = exts->KHR_timeline_semaphore;

   return VK_SUCCESS;
}

static void
wrapper_apply_device_extension_blacklist(struct wrapper_physical_device *physical_device) {
   const char *env = getenv("WRAPPER_EXTENSION_BLACKLIST");
   char *blacklist, *save = NULL;

   if (!env)
      return;

   /* strtok() would chop the environment variable itself apart */
   blacklist = strdup(env);
   if (!blacklist)
      return;

   for (char *extension = strtok_r(blacklist, ",", &save); extension != NULL;
        extension = strtok_r(NULL, ",", &save)) {
      while (*extension == ' ')
         extension++;

      for (int i = 0; i < VK_DEVICE_EXTENSION_COUNT; i++) {
         /* exact match: "VK_KHR_swapchain" must not also drop
          * "VK_KHR_swapchain_mutable_format" */
         if (strcmp(extension, vk_device_extensions[i].extensionName) == 0) {
            WRAPPER_LOG(info, "Blacklisting extension %s", extension);
            physical_device->vk.supported_extensions.extensions[i] = false;
         }
      }
   }

   free(blacklist);
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
wrapper_wsi_proc_addr(VkPhysicalDevice physicalDevice, const char *pName)
{
   VK_FROM_HANDLE(vk_physical_device, pdevice, physicalDevice);
   return vk_instance_get_proc_addr_unchecked(pdevice->instance, pName);
}

VkResult enumerate_physical_device(struct vk_instance *_instance)
{
   struct wrapper_instance *instance = (struct wrapper_instance *)_instance;
   VkPhysicalDevice *physical_devices;
   uint32_t physical_device_count = 0;
   static int wrapper_disable_placed = -1;
   static int wrapper_disable_present_wait = -1;
   VkResult result;

   result = instance->dispatch_table.EnumeratePhysicalDevices(
      instance->dispatch_handle, &physical_device_count, NULL);

   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to retrieve physical devices count, res %d", result);
      return result;
   }

   if (physical_device_count == 0)
      return VK_SUCCESS;

   physical_devices = malloc(physical_device_count * sizeof(VkPhysicalDevice));
   if (!physical_devices)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   result = instance->dispatch_table.EnumeratePhysicalDevices(
      instance->dispatch_handle, &physical_device_count, physical_devices);

   if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
      WRAPPER_LOG(error, "Failed to enumerate physical devices, res %d", result);
      free(physical_devices);
      return result;
   }

   for (uint32_t i = 0; i < physical_device_count; i++) {
      PFN_vkGetInstanceProcAddr get_instance_proc_addr;
      struct wrapper_physical_device *pdevice;

      pdevice = vk_zalloc(&_instance->alloc, sizeof(*pdevice), 8,
                          VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
      if (!pdevice) {
         free(physical_devices);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      pdevice->drm.fd = -1;

      struct vk_physical_device_dispatch_table dispatch_table;
      vk_physical_device_dispatch_table_from_entrypoints(
         &dispatch_table, &wrapper_physical_device_entrypoints, true);
      vk_physical_device_dispatch_table_from_entrypoints(
         &dispatch_table, &wsi_physical_device_entrypoints, false);
      vk_physical_device_dispatch_table_from_entrypoints(
         &dispatch_table, &wrapper_physical_device_trampolines, false);

      result = vk_physical_device_init(&pdevice->vk,
                                       &instance->vk,
                                       NULL, NULL, NULL,
                                       &dispatch_table);
      if (result != VK_SUCCESS) {
         vk_free(&_instance->alloc, pdevice);
         free(physical_devices);
         return result;
      }

      pdevice->instance = instance;
      pdevice->dispatch_handle = physical_devices[i];
      get_instance_proc_addr = instance->dispatch_table.GetInstanceProcAddr;

      vk_physical_device_dispatch_table_load(&pdevice->dispatch_table,
                                             get_instance_proc_addr,
                                             instance->dispatch_handle);

      /* ---- identity: this wrapper only drives Arm Mali GPUs ---- */
      pdevice->driver_properties = (VkPhysicalDeviceDriverProperties) {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
      };
      pdevice->properties2 = (VkPhysicalDeviceProperties2) {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
         .pNext = &pdevice->driver_properties,
      };
      pdevice->dispatch_table.GetPhysicalDeviceProperties2(
         pdevice->dispatch_handle, &pdevice->properties2);

      if (!wrapper_is_mali_device(&pdevice->properties2.properties,
                                  &pdevice->driver_properties)) {
         WRAPPER_LOG(error, "Ignoring '%s' (vendor 0x%04x, driver id %d): this wrapper "
                     "only supports Arm Mali GPUs",
                     pdevice->properties2.properties.deviceName,
                     pdevice->properties2.properties.vendorID,
                     pdevice->driver_properties.driverID);
         vk_physical_device_finish(&pdevice->vk);
         vk_free(&_instance->alloc, pdevice);
         continue;
      }

      /* real GPU facts straight from the kernel (/dev/mali0) + a DRM node */
      wrapper_kbase_probe(&pdevice->mali);
      wrapper_mali_finish_info(&pdevice->mali,
                               pdevice->properties2.properties.deviceName,
                               pdevice->properties2.properties.deviceID);
      wrapper_drm_init(&pdevice->drm);

      if (wrapper_disable_placed == -1)
         wrapper_disable_placed = getenv("WRAPPER_DISABLE_PLACED") ? atoi(getenv("WRAPPER_DISABLE_PLACED")) : 0;

      if (wrapper_disable_present_wait == -1)
         wrapper_disable_present_wait = getenv("WRAPPER_DISABLE_PRESENT_WAIT") && atoi(getenv("WRAPPER_DISABLE_PRESENT_WAIT"));

      result = wrapper_setup_device_extensions(pdevice);
      if (result != VK_SUCCESS) {
         WRAPPER_LOG(error, "Failed to query device extensions, res %d", result);
         wrapper_drm_finish(&pdevice->drm);
         vk_physical_device_finish(&pdevice->vk);
         vk_free(&_instance->alloc, pdevice);
         free(physical_devices);
         return result;
      }
      wrapper_apply_device_extension_blacklist(pdevice);
      wrapper_setup_device_features(pdevice);

      struct vk_features *supported_features = &pdevice->vk.supported_features;
      pdevice->base_supported_features = *supported_features;
      supported_features->presentId = true;
      supported_features->multiViewport = true;
      supported_features->depthClamp = true;
      supported_features->depthBiasClamp = true;
      if (!wrapper_disable_placed) {
        pdevice->vk.supported_extensions.EXT_map_memory_placed = true;
        pdevice->vk.supported_extensions.KHR_map_memory2 = true;
      	supported_features->memoryMapPlaced = true;
      	supported_features->memoryUnmapReserve = true;
      } else {
        WRAPPER_LOG(info, "Disabling VK_EXT_map_memory_placed");
      	pdevice->vk.supported_extensions.EXT_map_memory_placed = false;
        pdevice->vk.supported_extensions.KHR_map_memory2 = false;
      	supported_features->memoryMapPlaced = false;
      	supported_features->memoryUnmapReserve = false;
      }
      /* textureCompressionBC is reported exactly as the Mali driver reports it */
      supported_features->fillModeNonSolid = true;
      supported_features->shaderClipDistance = true;
      supported_features->shaderCullDistance = true;
      if (wrapper_disable_present_wait) {
         WRAPPER_LOG(info, "Disabling present wait");
         supported_features->presentWait = false;
         pdevice->vk.supported_extensions.KHR_present_wait = false;
      } else {
         supported_features->presentWait = supported_features->timelineSemaphore;
      }
      supported_features->swapchainMaintenance1 = true;
      supported_features->imageCompressionControlSwapchain = false;
      
      result = wsi_device_init(&pdevice->wsi_device,
                               wrapper_physical_device_to_handle(pdevice),
                               wrapper_wsi_proc_addr, &_instance->alloc, -1,
                               NULL, &(struct wsi_device_options){});
      if (result != VK_SUCCESS) {
         wrapper_drm_finish(&pdevice->drm);
         vk_physical_device_finish(&pdevice->vk);
         vk_free(&_instance->alloc, pdevice);
         free(physical_devices);
         return result;
      }
      pdevice->vk.wsi_device = &pdevice->wsi_device;
      pdevice->wsi_device.force_bgra8_unorm_first = true;

      pdevice->dispatch_table.GetPhysicalDeviceMemoryProperties(
         pdevice->dispatch_handle, &pdevice->memory_properties);

      char driver_version_str[48];
      format_driver_version(pdevice->properties2.properties.driverVersion,
                            driver_version_str, sizeof(driver_version_str));
      WRAPPER_LOG(info, "GPU Name: %s (vendor 0x%04x device 0x%08x)",
                  pdevice->properties2.properties.deviceName,
                  pdevice->properties2.properties.vendorID,
                  pdevice->properties2.properties.deviceID);
      WRAPPER_LOG(info, "Driver Version: %s", driver_version_str);

      const char *engine_name = instance->vk.app_info.engine_name
         ? instance->vk.app_info.engine_name : "wrapper";
      pdevice->wsi_device.engine_name = engine_name;

      const uint32_t engine_version = instance->vk.app_info.engine_version;

      /* Arm Mali proprietary driver workarounds */
      
      if (pdevice->driver_properties.driverID == VK_DRIVER_ID_ARM_PROPRIETARY) {
         bool is_dxvk = strstr(engine_name, "DXVK");
         bool is_vkd3d = strstr(engine_name, "vkd3d");
         bool is_d3d = is_dxvk || is_vkd3d;
         pdevice->is_vkd3d = is_vkd3d;
         if (is_d3d) {
            WRAPPER_LOG(info, "Faking VK_EXT_robustness2 for engine '%s'", engine_name);
            pdevice->vk.supported_extensions.EXT_robustness2 = true;
            supported_features->robustBufferAccess2 = true;
            supported_features->nullDescriptor = true;
            if (is_dxvk && engine_version >= VK_MAKE_VERSION(2, 7, 0)) {
               WRAPPER_LOG(info, "Faking VK_KHR_pipeline_library");
               pdevice->vk.supported_extensions.KHR_pipeline_library = true;
            }
            supported_features->extendedDynamicState = true;
            supported_features->extendedDynamicState2 = true;
            supported_features->dualSrcBlend = true;
            supported_features->multiDrawIndirect = true;
         }
         /* Spoof both vertex_attribute_divisor extensions as supported. Mali r51
          * exposes KHR (we forward that); r44 exposes neither, so we advertise
          * it purely to get vkd3d past its requirement check. Advertising BOTH
          * (rather than hiding KHR) is fine -- the device-side filter below
          * forwards whichever alias the base driver actually has, and
          * process_pnext_chain aliases the feature struct EXT<->KHR to match. */
         WRAPPER_LOG(info, "Spoofing VK_EXT/KHR_vertex_attribute_divisor as supported");
         pdevice->vk.supported_extensions.EXT_vertex_attribute_divisor = true;
         pdevice->vk.supported_extensions.KHR_vertex_attribute_divisor = true;
         WRAPPER_LOG(info, "Disabling VK_EXT_calibrated_timestamps");
         pdevice->vk.supported_extensions.EXT_calibrated_timestamps = false;
         if (!is_d3d) {
            WRAPPER_LOG(info, "Disabling VK_EXT_extended_dynamic_state and VK_EXT_extended_dynamic_state2");
            pdevice->vk.supported_extensions.EXT_extended_dynamic_state = false;
            pdevice->vk.supported_extensions.EXT_extended_dynamic_state2 = false;
         }
      }

      /* DXVK > 2.4.1 requires VK_KHR_maintenance5. Some Mali drivers (e.g. r44)
       * don't expose it. Emulate it for any D3D client whose base driver lacks
       * it (NOT Samsung-gated -- the base-lacks guard limits it to drivers that
       * actually need it). Feature faked in Features2, struct unlinked from the
       * real CreateDevice, entrypoints implemented in wrapper_device.c. */
      {
         bool is_d3d = strstr(engine_name, "DXVK") || strstr(engine_name, "vkd3d");
         if (is_d3d && !pdevice->base_supported_extensions.KHR_maintenance5) {
            WRAPPER_LOG(info, "Faking VK_KHR_maintenance5 (base driver lacks it)");
            pdevice->vk.supported_extensions.KHR_maintenance5 = true;
         }
      }

      /* VK_KHR_push_descriptor: vkd3d/DXVK effectively require it, but some
       * drivers (e.g. Mali r44) don't expose it. Advertise + emulate it for D3D
       * clients whose base driver lacks it. maxPushDescriptors is reported in
       * wrapper_GetPhysicalDeviceProperties2; the entrypoints are emulated in
       * wrapper_device.c. */
      {
         bool is_d3d = strstr(engine_name, "DXVK") || strstr(engine_name, "vkd3d");
         if (is_d3d && !pdevice->base_supported_extensions.KHR_push_descriptor) {
            WRAPPER_LOG(info, "Faking VK_KHR_push_descriptor (base driver lacks it)");
            pdevice->vk.supported_extensions.KHR_push_descriptor = true;
         }
      }

      list_addtail(&pdevice->vk.link, &_instance->physical_devices.list);
   }

   free(physical_devices);
   return VK_SUCCESS;
}

void destroy_physical_device(struct vk_physical_device *pdevice) {
   VK_FROM_HANDLE(wrapper_physical_device, wpdevice,
                  vk_physical_device_to_handle(pdevice));
   wrapper_drm_finish(&wpdevice->drm);
   wsi_device_finish(pdevice->wsi_device, &pdevice->instance->alloc);
   vk_physical_device_finish(pdevice);
   vk_free(&pdevice->instance->alloc, pdevice);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_EnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                           const char* pLayerName,
                                           uint32_t* pPropertyCount,
                                           VkExtensionProperties* pProperties)
{
   return vk_common_EnumerateDeviceExtensionProperties(physicalDevice,
                                                       pLayerName,
                                                       pPropertyCount,
                                                       pProperties);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetPhysicalDeviceFeatures(VkPhysicalDevice physicalDevice,
                                  VkPhysicalDeviceFeatures* pFeatures) 
{
   return vk_common_GetPhysicalDeviceFeatures(physicalDevice, pFeatures);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetPhysicalDeviceFeatures2(VkPhysicalDevice physicalDevice,
                                   VkPhysicalDeviceFeatures2* pFeatures) {
   VK_FROM_HANDLE(wrapper_physical_device, pdevice, physicalDevice);
   vk_common_GetPhysicalDeviceFeatures2(physicalDevice, pFeatures);

   if (pdevice->driver_properties.driverID == VK_DRIVER_ID_ARM_PROPRIETARY &&
       pdevice->vk.supported_extensions.EXT_robustness2) {
      vk_foreach_struct(s, pFeatures->pNext) {
         if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT) {
            VkPhysicalDeviceRobustness2FeaturesEXT *r2 =
               (VkPhysicalDeviceRobustness2FeaturesEXT *)s;
            r2->robustBufferAccess2 = VK_TRUE;
            r2->nullDescriptor = VK_TRUE;
            if (pdevice->is_vkd3d)
               r2->robustImageAccess2 = VK_TRUE;
         }
         if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT) {
            ((VkPhysicalDeviceExtendedDynamicStateFeaturesEXT *)s)->extendedDynamicState = VK_TRUE;
         }
         if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT) {
            ((VkPhysicalDeviceExtendedDynamicState2FeaturesEXT *)s)->extendedDynamicState2 = VK_TRUE;
         }
         if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT) {
            VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT *vad =
               (VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT *)s;
            vad->vertexAttributeInstanceRateDivisor = VK_TRUE;
            vad->vertexAttributeInstanceRateZeroDivisor = VK_TRUE;
         }
      }
   }

   /* VK_KHR_maintenance5 is emulated by the wrapper for D3D clients when the
    * Mali driver lacks it (see enumerate_physical_device): report the feature
    * as available whenever the extension is advertised-but-faked, otherwise
    * DXVK 2.5+ rejects the device. */
   if (pdevice->vk.supported_extensions.KHR_maintenance5 &&
       !pdevice->base_supported_extensions.KHR_maintenance5) {
      vk_foreach_struct(s, pFeatures->pNext) {
         if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES)
            ((VkPhysicalDeviceMaintenance5Features *)s)->maintenance5 = VK_TRUE;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetPhysicalDeviceProperties(VkPhysicalDevice physicalDevice,
                                    VkPhysicalDeviceProperties *pProperties)
{
   uint32_t api_version = parse_vk_version_from_env();

   VK_FROM_HANDLE(wrapper_physical_device, pdevice, physicalDevice);
   pdevice->dispatch_table.GetPhysicalDeviceProperties(
      pdevice->dispatch_handle, pProperties);

   /* deviceName, vendorID and deviceID are the real values the Mali driver
    * reports. Only if the driver left deviceID empty, use kbase's GPU_ID. */
   if (pProperties->deviceID == 0 && pdevice->mali.gpu_id != 0)
      pProperties->deviceID = pdevice->mali.gpu_id;

   if (api_version > 0)
      pProperties->apiVersion = api_version;
}

/* vkd3d's bindless model needs large UpdateAfterBind descriptor limits (D3D12
 * shader-visible heaps hold up to ~1M descriptors). Some drivers (e.g. Mali
 * r44) report only ~500k, so vkd3d aborts at bindless_state_init with
 * "Insufficient descriptor indexing support". Spoof the limits up to the D3D12
 * heap size for D3D clients. NOTE: unlike a feature toggle this is a REAL
 * hardware limit -- if the game actually binds more descriptors than the driver
 * can back, it will fault later (VK_EXT_device_fault will report where) rather
 * than being cleanly rejected here. Only bumps values that are below target. */
#define WRAPPER_D3D_BINDLESS_LIMIT (1u << 20)   /* 1048576 */
#define BUMP_UAB(p, t) do { \
   uint32_t _t = (t); \
   if ((p)->maxUpdateAfterBindDescriptorsInAllPools < _t)           (p)->maxUpdateAfterBindDescriptorsInAllPools = _t; \
   if ((p)->maxPerStageDescriptorUpdateAfterBindSamplers < _t)      (p)->maxPerStageDescriptorUpdateAfterBindSamplers = _t; \
   if ((p)->maxPerStageDescriptorUpdateAfterBindUniformBuffers < _t)(p)->maxPerStageDescriptorUpdateAfterBindUniformBuffers = _t; \
   if ((p)->maxPerStageDescriptorUpdateAfterBindStorageBuffers < _t)(p)->maxPerStageDescriptorUpdateAfterBindStorageBuffers = _t; \
   if ((p)->maxPerStageDescriptorUpdateAfterBindSampledImages < _t) (p)->maxPerStageDescriptorUpdateAfterBindSampledImages = _t; \
   if ((p)->maxPerStageDescriptorUpdateAfterBindStorageImages < _t) (p)->maxPerStageDescriptorUpdateAfterBindStorageImages = _t; \
   if ((p)->maxPerStageUpdateAfterBindResources < _t)              (p)->maxPerStageUpdateAfterBindResources = _t; \
   if ((p)->maxDescriptorSetUpdateAfterBindSamplers < _t)          (p)->maxDescriptorSetUpdateAfterBindSamplers = _t; \
   if ((p)->maxDescriptorSetUpdateAfterBindUniformBuffers < _t)    (p)->maxDescriptorSetUpdateAfterBindUniformBuffers = _t; \
   if ((p)->maxDescriptorSetUpdateAfterBindStorageBuffers < _t)    (p)->maxDescriptorSetUpdateAfterBindStorageBuffers = _t; \
   if ((p)->maxDescriptorSetUpdateAfterBindSampledImages < _t)     (p)->maxDescriptorSetUpdateAfterBindSampledImages = _t; \
   if ((p)->maxDescriptorSetUpdateAfterBindStorageImages < _t)     (p)->maxDescriptorSetUpdateAfterBindStorageImages = _t; \
} while (0)

VKAPI_ATTR void VKAPI_CALL
wrapper_GetPhysicalDeviceProperties2(VkPhysicalDevice physicalDevice,
                                     VkPhysicalDeviceProperties2* pProperties)
{
   uint32_t api_version = parse_vk_version_from_env();

   VK_FROM_HANDLE(wrapper_physical_device, pdevice, physicalDevice);
   pdevice->dispatch_table.GetPhysicalDeviceProperties2(
      pdevice->dispatch_handle, pProperties);

   const char *eng = pdevice->instance->vk.app_info.engine_name;
   bool is_d3d = eng && (strstr(eng, "DXVK") || strstr(eng, "vkd3d"));

   /* real deviceName / vendorID / deviceID / driver identity pass through; only
    * an empty deviceID is completed from kbase's GPU_ID */
   if (pProperties->properties.deviceID == 0 && pdevice->mali.gpu_id != 0)
      pProperties->properties.deviceID = pdevice->mali.gpu_id;

   if (api_version > 0)
      pProperties->properties.apiVersion = api_version;

   vk_foreach_struct(prop, pProperties->pNext) {
      switch (prop->sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES_EXT:
      {
         VkPhysicalDeviceVertexAttributeDivisorPropertiesKHR khr_vad = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES_KHR,
         };
         VkPhysicalDeviceProperties2 p2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &khr_vad,
         };
         pdevice->dispatch_table.GetPhysicalDeviceProperties2(
            pdevice->dispatch_handle, &p2);
         ((VkPhysicalDeviceVertexAttributeDivisorPropertiesEXT *)prop)->maxVertexAttribDivisor =
            khr_vad.maxVertexAttribDivisor;
         break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_PROPERTIES_EXT:
      {
         VkPhysicalDeviceMapMemoryPlacedPropertiesEXT *placed_prop =
               (VkPhysicalDeviceMapMemoryPlacedPropertiesEXT *)prop;
         uint64_t os_page_size;
         os_get_page_size(&os_page_size);
         placed_prop->minPlacedMemoryMapAlignment = os_page_size;
         break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES_EXT:
      {
      	 VkPhysicalDeviceTexelBufferAlignmentPropertiesEXT *texel_prop =
      	      (VkPhysicalDeviceTexelBufferAlignmentPropertiesEXT *)prop;
      	 texel_prop->storageTexelBufferOffsetAlignmentBytes = 1;
      	 texel_prop->uniformTexelBufferOffsetAlignmentBytes = 1;
      	 break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES:
      {
         /* Only override when we're faking the extension; if the base driver has
          * it natively it already filled a real limit. */
         if (pdevice->vk.supported_extensions.KHR_push_descriptor &&
             !pdevice->base_supported_extensions.KHR_push_descriptor)
            ((VkPhysicalDevicePushDescriptorProperties *)prop)->maxPushDescriptors =
               WRAPPER_MAX_PUSH_DESCRIPTORS;
         break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES:
      {
         VkPhysicalDeviceVulkan11Properties *vk11_prop =
              (VkPhysicalDeviceVulkan11Properties *)prop;
         if (!wrapper_subgroup_ops_passthrough()) {
            vk11_prop->subgroupSupportedOperations = 0;
            vk11_prop->subgroupSupportedStages = 0;
         }
         break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES:
      {
         VkPhysicalDeviceVulkan12Properties *vk12_prop =
              (VkPhysicalDeviceVulkan12Properties *)prop;
         /* driverID / driverName / driverInfo stay exactly as the Mali driver reports */
         /* Vulkan12Properties folds in the descriptor-indexing limits. */
         if (is_d3d)
            BUMP_UAB(vk12_prop, WRAPPER_D3D_BINDLESS_LIMIT);
         break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES:
      {
         if (is_d3d)
            BUMP_UAB((VkPhysicalDeviceDescriptorIndexingProperties *)prop,
                     WRAPPER_D3D_BINDLESS_LIMIT);
         break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES:
      {
         VkPhysicalDeviceVulkan13Properties *vk13_prop =
              (VkPhysicalDeviceVulkan13Properties *)prop;
         vk13_prop->storageTexelBufferOffsetAlignmentBytes = 1;
         vk13_prop->uniformTexelBufferOffsetAlignmentBytes = 1;
         break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES:
      {
         VkPhysicalDeviceSubgroupProperties *subgroup_prop =
              (VkPhysicalDeviceSubgroupProperties *)prop;
         if (!wrapper_subgroup_ops_passthrough()) {
            subgroup_prop->supportedOperations = 0;
            subgroup_prop->supportedStages = 0;
         }
         break;
      }
      default:
         break;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
wrapper_GetPhysicalDeviceMemoryProperties(VkPhysicalDevice physicalDevice,
										  VkPhysicalDeviceMemoryProperties *pMemoryProperties)
{
   VK_FROM_HANDLE(wrapper_physical_device, pdevice, physicalDevice);

   static int wrapper_vmem_max_size = -1;
   
   pdevice->dispatch_table.GetPhysicalDeviceMemoryProperties(
      pdevice->dispatch_handle, pMemoryProperties);

   if (wrapper_vmem_max_size == -1)
      wrapper_vmem_max_size = getenv("WRAPPER_VMEM_MAX_SIZE") ? atoi(getenv("WRAPPER_VMEM_MAX_SIZE")) : 0;

   if (wrapper_vmem_max_size > 0)
      pMemoryProperties->memoryHeaps[0].size = (VkDeviceSize)wrapper_vmem_max_size * 1048576;   
}

VKAPI_ATTR void VKAPI_CALL                                                                                      
wrapper_GetPhysicalDeviceMemoryProperties2(VkPhysicalDevice physicalDevice,
                                           VkPhysicalDeviceMemoryProperties2 *pMemoryProperties)
{
   VK_FROM_HANDLE(wrapper_physical_device, pdevice, physicalDevice);

   static int wrapper_vmem_max_size = -1;

   pdevice->dispatch_table.GetPhysicalDeviceMemoryProperties2(
      pdevice->dispatch_handle, pMemoryProperties);

   if (wrapper_vmem_max_size == -1)
      wrapper_vmem_max_size = getenv("WRAPPER_VMEM_MAX_SIZE") ? atoi(getenv("WRAPPER_VMEM_MAX_SIZE")) : 0;

   if (wrapper_vmem_max_size > 0)
      pMemoryProperties->memoryProperties.memoryHeaps[0].size = (VkDeviceSize)wrapper_vmem_max_size * 1048576;
}
