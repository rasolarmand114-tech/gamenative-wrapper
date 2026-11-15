/*
 * Host-visible memory with VK_EXT_map_memory_placed emulation.
 *
 * Backend: AHardwareBuffer (gralloc) only. The memory is allocated by the
 * system driver as an exportable AHardwareBuffer, so the CPU mapping that the
 * application asked for and the GPU allocation are the same pages (zero-copy,
 * no staging, no /dev/dma_heap or /dev/ion access that SELinux denies to apps).
 *
 * Only when a resource explicitly forbids AHB handles (its external handle
 * types are dma-buf only) the DRM node is used as an allocator of dma-bufs that
 * are imported into the driver. When neither works the allocation is simply
 * forwarded to the driver (it just cannot be placed-mapped).
 */

#include "wrapper_private.h"
#include "wrapper_log.h"
#include "wrapper_entrypoints.h"
#include "vk_common_entrypoints.h"
#include "util/os_file.h"
#include "vk_util.h"

#include <android/hardware_buffer.h>
#include <vndk/hardware_buffer.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#define WRAPPER_HANDLE_AHB \
   VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID
#define WRAPPER_HANDLE_DMABUF VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT

static uint32_t
wrapper_select_allowed_device_memory_type(struct wrapper_device *device,
                                          uint32_t allowed_type_bits,
                                          VkMemoryPropertyFlags flags,
                                          VkMemoryPropertyFlags minimum_flags)
{
   const VkPhysicalDeviceMemoryProperties *props =
      &device->physical->memory_properties;

   /* best match first (all requested flags), then the minimum we can live with */
   for (uint32_t pass = 0; pass < 2; pass++) {
      VkMemoryPropertyFlags want = pass == 0 ? flags : minimum_flags;

      for (uint32_t idx = 0; idx < props->memoryTypeCount; idx++) {
         if (!(allowed_type_bits & (1U << idx)))
            continue;
         if ((props->memoryTypes[idx].propertyFlags & want) == want)
            return idx;
      }
   }
   return UINT32_MAX;
}

/* Temporarily removes one struct from a pNext chain and puts it back later, so
 * that a failed emulation can still hand the *unmodified* application chain to
 * the driver (and the application's memory is never left modified). */
struct wrapper_pnext_unlink {
   VkBaseOutStructure *prev;    /* NULL when the struct was the chain head */
   VkBaseOutStructure *removed;
};

static void
wrapper_pnext_unlink(VkMemoryAllocateInfo *info, VkStructureType sType,
                     struct wrapper_pnext_unlink *u)
{
   VkBaseOutStructure *prev = NULL;

   u->prev = NULL;
   u->removed = NULL;

   for (VkBaseOutStructure *s = (VkBaseOutStructure *)info->pNext; s;
        prev = s, s = s->pNext) {
      if (s->sType != sType)
         continue;

      u->prev = prev;
      u->removed = s;
      if (prev)
         prev->pNext = s->pNext;      /* application owned memory */
      else
         info->pNext = s->pNext;      /* our local copy of the info */
      return;
   }
}

static void
wrapper_pnext_relink(struct wrapper_pnext_unlink *u)
{
   if (u->prev && u->removed)
      u->prev->pNext = u->removed;
   u->prev = NULL;
   u->removed = NULL;
}

static VkResult
wrapper_allocate_memory_ahardware_buffer(struct wrapper_device *device,
                                         const VkMemoryAllocateInfo *pAllocateInfo,
                                         const VkAllocationCallbacks *pAllocator,
                                         VkDeviceMemory *pMemory,
                                         AHardwareBuffer **pAHardwareBuffer)
{
   VkExportMemoryAllocateInfo export_memory_info;
   VkMemoryAllocateInfo allocate_info;
   VkResult result;

   const VkMemoryDedicatedAllocateInfo *memory_dedicated_info =
      vk_find_struct_const(pAllocateInfo->pNext, MEMORY_DEDICATED_ALLOCATE_INFO);

   if (!device->dispatch_table.GetMemoryAndroidHardwareBufferANDROID)
      return VK_ERROR_EXTENSION_NOT_PRESENT;

   export_memory_info = (VkExportMemoryAllocateInfo) {
      .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
      .pNext = pAllocateInfo->pNext,
      .handleTypes = WRAPPER_HANDLE_AHB,
   };
   allocate_info = *pAllocateInfo;
   allocate_info.pNext = &export_memory_info;

   if (memory_dedicated_info && memory_dedicated_info->image != VK_NULL_HANDLE) {
      /* An AHB exported for a dedicated image takes its size from the image */
      WRAPPER_LOG(info, "Dedicated image allocation, letting the driver size the AHB");
      allocate_info.allocationSize = 0;
   }

   result = device->dispatch_table.AllocateMemory(device->dispatch_handle,
                                                  &allocate_info,
                                                  pAllocator,
                                                  pMemory);
   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to allocate ahb memory, res %d", result);
      return result;
   }

   result = device->dispatch_table.GetMemoryAndroidHardwareBufferANDROID(
      device->dispatch_handle,
      &(VkMemoryGetAndroidHardwareBufferInfoANDROID) {
         .sType =
            VK_STRUCTURE_TYPE_MEMORY_GET_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
         .memory = *pMemory,
      },
      pAHardwareBuffer);

   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to export ahb, res %d", result);
      return result;
   }

   if (AHardwareBuffer_getNativeHandle(*pAHardwareBuffer) == NULL) {
      WRAPPER_LOG(error, "Invalid AHB native handle");
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   }

   return VK_SUCCESS;
}

/* dma-buf from the DRM node, imported into the driver. Only used for resources
 * that can't be backed by an AHB. */
static VkResult
wrapper_allocate_memory_drm_dmabuf(struct wrapper_device *device,
                                   const VkMemoryAllocateInfo *pAllocateInfo,
                                   const VkAllocationCallbacks *pAllocator,
                                   VkDeviceMemory *pMemory,
                                   int *out_fd)
{
   VkImportMemoryFdInfoKHR import_fd_info;
   VkMemoryAllocateInfo allocate_info;
   VkResult result;

   if (!device->dispatch_table.GetMemoryFdPropertiesKHR)
      return VK_ERROR_EXTENSION_NOT_PRESENT;

   *out_fd = wrapper_drm_alloc_dmabuf(&device->physical->drm,
                                      pAllocateInfo->allocationSize);
   if (*out_fd < 0)
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;

   VkMemoryFdPropertiesKHR memory_fd_props = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
   };
   result = device->dispatch_table.GetMemoryFdPropertiesKHR(
      device->dispatch_handle, WRAPPER_HANDLE_DMABUF, *out_fd, &memory_fd_props);
   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to get memory fd properties, res %d", result);
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   }

   uint32_t memory_type_index = wrapper_select_allowed_device_memory_type(device,
      memory_fd_props.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
   if (memory_type_index == UINT32_MAX) {
      WRAPPER_LOG(error, "No compatible memory type found for fd %d", *out_fd);
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   }

   import_fd_info = (VkImportMemoryFdInfoKHR) {
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
      .pNext = pAllocateInfo->pNext,
      .fd = os_dupfd_cloexec(*out_fd),
      .handleType = WRAPPER_HANDLE_DMABUF,
   };
   if (import_fd_info.fd < 0)
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;

   allocate_info = *pAllocateInfo;
   allocate_info.pNext = &import_fd_info;
   allocate_info.memoryTypeIndex = memory_type_index;

   result = device->dispatch_table.AllocateMemory(device->dispatch_handle,
                                                  &allocate_info, pAllocator, pMemory);
   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "Failed to import DRM dma-buf memory, res %d", result);
      close(import_fd_info.fd);   /* only consumed by the driver on success */
   }

   return result;
}

static void
wrapper_device_memory_reset(struct wrapper_device_memory *mem,
                            const VkAllocationCallbacks *driver_alloc)
{
   struct wrapper_device *device = mem->device;

   if (mem->ahardware_buffer) {
      AHardwareBuffer_release(mem->ahardware_buffer);
      mem->ahardware_buffer = NULL;
   }
   if (mem->fd != -1) {
      close(mem->fd);
      mem->fd = -1;
   }
   if (mem->map_address && mem->map_size) {
      munmap(mem->map_address, mem->map_size);
   }
   mem->map_address = NULL;
   mem->map_size = 0;
   if (mem->dispatch_handle != VK_NULL_HANDLE) {
      device->dispatch_table.FreeMemory(device->dispatch_handle,
                                        mem->dispatch_handle, driver_alloc);
      mem->dispatch_handle = VK_NULL_HANDLE;
   }
}

VkResult
wrapper_device_memory_create(struct wrapper_device *device,
                             const VkAllocationCallbacks *alloc,
                             struct wrapper_device_memory **out_mem)
{
   struct wrapper_device_memory *mem =
      vk_zalloc2(&device->vk.alloc, alloc, sizeof(*mem), 8,
                 VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);

   if (mem == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   mem->fd = -1;
   mem->device = device;
   /* keep our own copy so a later teardown never touches the application's
    * (possibly already gone) callback structure */
   if (alloc) {
      mem->alloc_copy = *alloc;
      mem->has_alloc = true;
   }
   *out_mem = mem;
   return VK_SUCCESS;
}

/* `mem` must not be registered (or already removed from the lists). */
void
wrapper_device_memory_destroy(struct wrapper_device_memory *mem,
                              const VkAllocationCallbacks *alloc)
{
   struct wrapper_device *device = mem->device;

   wrapper_device_memory_reset(mem, alloc);
   vk_free2(&device->vk.alloc, alloc, mem);
}

void
wrapper_device_memory_destroy_all(struct wrapper_device *device)
{
   simple_mtx_lock(&device->resource_mutex);

   list_for_each_entry_safe(struct wrapper_device_memory, mem,
                            &device->device_memory_list, link) {
      const VkAllocationCallbacks *alloc = mem->has_alloc ? &mem->alloc_copy : NULL;

      list_del(&mem->link);
      if (mem->dispatch_handle != VK_NULL_HANDLE)
         _mesa_hash_table_u64_remove(device->memory_table,
                                     (uint64_t)mem->dispatch_handle);
      wrapper_device_memory_destroy(mem, alloc);
   }

   simple_mtx_unlock(&device->resource_mutex);
}

static void
wrapper_device_memory_register(struct wrapper_device *device,
                               struct wrapper_device_memory *mem)
{
   simple_mtx_lock(&device->resource_mutex);
   list_add(&mem->link, &device->device_memory_list);
   _mesa_hash_table_u64_insert(device->memory_table,
                               (uint64_t)mem->dispatch_handle, mem);
   simple_mtx_unlock(&device->resource_mutex);
}

static struct wrapper_device_memory *
wrapper_device_memory_from_handle(struct wrapper_device *device,
                                  VkDeviceMemory handle)
{
   struct wrapper_device_memory *mem;

   if (handle == VK_NULL_HANDLE)
      return NULL;

   simple_mtx_lock(&device->resource_mutex);
   mem = _mesa_hash_table_u64_search(device->memory_table, (uint64_t)handle);
   simple_mtx_unlock(&device->resource_mutex);

   return mem;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_AllocateMemory(VkDevice _device,
                       const VkMemoryAllocateInfo *pAllocateInfo,
                       const VkAllocationCallbacks *pAllocator,
                       VkDeviceMemory *pMemory)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   struct wrapper_device_memory *mem = NULL;
   struct wrapper_pnext_unlink unlinked = { 0 };
   VkResult result;

   VkMemoryPropertyFlags property_flags =
      device->physical->memory_properties.memoryTypes[
         pAllocateInfo->memoryTypeIndex].propertyFlags;

   if (!(property_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
      goto fallback;

   if (!device->vk.enabled_features.memoryMapPlaced ||
       !device->vk.enabled_extensions.EXT_map_memory_placed)
      goto fallback;

   if (vk_find_struct_const(pAllocateInfo, IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID) ||
       vk_find_struct_const(pAllocateInfo, IMPORT_MEMORY_FD_INFO_KHR) ||
       vk_find_struct_const(pAllocateInfo, EXPORT_MEMORY_ALLOCATE_INFO))
      goto fallback;

   const VkMemoryDedicatedAllocateInfo *dedicated_allocate_info =
      vk_find_struct_const(pAllocateInfo->pNext, MEMORY_DEDICATED_ALLOCATE_INFO);

   /* What may the backing memory be exported as? (0 = unconstrained) */
   VkExternalMemoryHandleTypeFlags valid_handle_types = 0;
   bool is_wsi_image = false;

   if (dedicated_allocate_info) {
      simple_mtx_lock(&device->resource_mutex);
      /* buffer and image are mutually exclusive */
      if (dedicated_allocate_info->image != VK_NULL_HANDLE) {
         struct wrapper_image *img = get_wrapper_image_from_handle_locked(
            device, dedicated_allocate_info->image);
         if (img) {
            valid_handle_types |= img->handle_types;
            is_wsi_image = img->is_wsi_image;
         }
      }
      if (dedicated_allocate_info->buffer != VK_NULL_HANDLE) {
         struct wrapper_buffer *buf = get_wrapper_buffer_from_handle_locked(
            device, dedicated_allocate_info->buffer);
         if (buf)
            valid_handle_types |= buf->handle_types;
      }
      simple_mtx_unlock(&device->resource_mutex);
   }

   static int bypass_swapchains = -1;
   if (bypass_swapchains == -1)
      bypass_swapchains = getenv("WRAPPER_BYPASS_SWAPCHAIN_PLACED") ?
                          atoi(getenv("WRAPPER_BYPASS_SWAPCHAIN_PLACED")) : 0;

   if (bypass_swapchains && is_wsi_image) {
      WRAPPER_LOG(info, "Bypassing EXT_map_memory_placed emulation for swapchain image");
      goto fallback;
   }

#define VALID_HANDLE(type) (valid_handle_types == 0 || ((type) & valid_handle_types) != 0)

   /* AHB is the backend; the dma-buf/DRM tier only exists for resources that
    * forbid AHB. If nothing can serve the resource, don't emulate at all. */
   if (!VALID_HANDLE(WRAPPER_HANDLE_AHB) &&
       !(VALID_HANDLE(WRAPPER_HANDLE_DMABUF) && device->physical->drm.fd >= 0))
      goto fallback;

   WRAPPER_LOG(info, "Emulating vkAllocateMemory");

   VkMemoryAllocateInfo memory_allocate_info = *pAllocateInfo;
   if (dedicated_allocate_info && valid_handle_types == 0) {
      /* Driver "bug" on some mobile drivers: an empty dedicated-allocation hint
       * combined with an Import/Export struct could crash the driver. */
      wrapper_pnext_unlink(&memory_allocate_info,
                           VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                           &unlinked);
   }

   result = wrapper_device_memory_create(device, pAllocator, &mem);
   if (result != VK_SUCCESS) {
      wrapper_pnext_relink(&unlinked);
      return vk_error(device, result);
   }
   mem->alloc_size = pAllocateInfo->allocationSize;

   result = VK_ERROR_INVALID_EXTERNAL_HANDLE;

   if (VALID_HANDLE(WRAPPER_HANDLE_AHB)) {
      WRAPPER_LOG(info, "Using AHardwareBuffer memory backend");
      result = wrapper_allocate_memory_ahardware_buffer(device,
         &memory_allocate_info, pAllocator, &mem->dispatch_handle,
         &mem->ahardware_buffer);
   }

   if (result != VK_SUCCESS && VALID_HANDLE(WRAPPER_HANDLE_DMABUF) &&
       device->physical->drm.fd >= 0) {
      WRAPPER_LOG(info, "Using DRM dma-buf memory backend");
      wrapper_device_memory_reset(mem, pAllocator);
      result = wrapper_allocate_memory_drm_dmabuf(device,
         &memory_allocate_info, pAllocator, &mem->dispatch_handle, &mem->fd);
   }
#undef VALID_HANDLE

   /* the application's pNext chain is whole again from here on */
   wrapper_pnext_relink(&unlinked);

   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "EXT_map_memory_placed emulation failed (res %d), "
                  "using the driver's own allocation%s", result,
                  is_wsi_image ? " for the swapchain image" : "");
      wrapper_device_memory_destroy(mem, pAllocator);
      goto fallback;
   }

   wrapper_device_memory_register(device, mem);
   *pMemory = mem->dispatch_handle;
   return VK_SUCCESS;

fallback:
   return device->dispatch_table.AllocateMemory(device->dispatch_handle,
      pAllocateInfo, pAllocator, pMemory);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_FreeMemory(VkDevice _device, VkDeviceMemory _memory,
                   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   struct wrapper_device_memory *mem;

   if (_memory == VK_NULL_HANDLE)
      return;

   simple_mtx_lock(&device->resource_mutex);
   mem = _mesa_hash_table_u64_search(device->memory_table, (uint64_t)_memory);
   if (mem) {
      list_del(&mem->link);
      _mesa_hash_table_u64_remove(device->memory_table, (uint64_t)_memory);
   }
   simple_mtx_unlock(&device->resource_mutex);

   if (mem) {
      wrapper_device_memory_destroy(mem, pAllocator);
      return;
   }

   device->dispatch_table.FreeMemory(device->dispatch_handle, _memory, pAllocator);
}

/* The AHB native handle may carry several fds (e.g. metadata pipes next to the
 * actual memory); pick the first seekable one that is large enough. */
static int
wrapper_memory_pick_fd(struct wrapper_device_memory *mem)
{
   const native_handle_t *handle;

   if (!mem->ahardware_buffer)
      return mem->fd;

   handle = AHardwareBuffer_getNativeHandle(mem->ahardware_buffer);
   if (!handle)
      return -1;

   for (int idx = 0; idx < handle->numFds; idx++) {
      off_t size = lseek(handle->data[idx], 0, SEEK_END);

      if (size < 0) {
         WRAPPER_LOG(error, "lseek failed on AHB fd (idx=%d, fd=%d): errno %d, trying next fd",
                     idx, handle->data[idx], errno);
         continue;
      }
      if ((size_t)size >= mem->alloc_size)
         return handle->data[idx];
   }

   WRAPPER_LOG(error, "No usable AHB fd with size >= alloc_size %zu", mem->alloc_size);
   return -1;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_MapMemory2KHR(VkDevice _device,
                      const VkMemoryMapInfoKHR *pMemoryMapInfo,
                      void **ppData)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   const VkMemoryMapPlacedInfoEXT *placed_info = NULL;
   struct wrapper_device_memory *mem;
   VkResult result = VK_ERROR_MEMORY_MAP_FAILED;
   const size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
   size_t length;
   int fd;

   if (pMemoryMapInfo->flags & VK_MEMORY_MAP_PLACED_BIT_EXT)
      placed_info = vk_find_struct_const(pMemoryMapInfo->pNext,
         MEMORY_MAP_PLACED_INFO_EXT);

   mem = wrapper_device_memory_from_handle(device, pMemoryMapInfo->memory);

   if (!placed_info) {
      /* ordinary map: the driver does it */
      return device->dispatch_table.MapMemory(device->dispatch_handle,
         pMemoryMapInfo->memory, pMemoryMapInfo->offset, pMemoryMapInfo->size,
         0, ppData);
   }

   if (!mem) {
      /* A placed mapping needs memory we own. Handing back a driver pointer
       * that is not the requested address would silently corrupt the caller,
       * so refuse and let it fall back to a normal map. */
      WRAPPER_LOG(error, "Placed map requested for memory without emulation backing");
      return vk_error(device, VK_ERROR_MEMORY_MAP_FAILED);
   }

   WRAPPER_LOG(info, "Emulating vkMapMemory2KHR");

   simple_mtx_lock(&device->resource_mutex);

   if (mem->map_address) {
      if (placed_info->pPlacedAddress != mem->map_address) {
         WRAPPER_LOG(error, "Placed address/mapped address mismatch");
         goto fail;
      }
      goto out;
   }

   if (pMemoryMapInfo->offset & (page_size - 1)) {
      WRAPPER_LOG(error, "Placed map offset 0x%llx is not page aligned",
                  (unsigned long long)pMemoryMapInfo->offset);
      goto fail;
   }

   fd = wrapper_memory_pick_fd(mem);
   if (fd < 0)
      goto fail;

   if (pMemoryMapInfo->size == VK_WHOLE_SIZE) {
      size_t total = mem->alloc_size;

      if (total == 0) {
         off_t res = lseek(fd, 0, SEEK_END);
         if (res < 0) {
            WRAPPER_LOG(error, "Failed lseek for file descriptor %d: errno %d", fd, errno);
            goto fail;
         }
         total = (size_t)res;
      }
      if (pMemoryMapInfo->offset >= total)
         goto fail;
      length = total - (size_t)pMemoryMapInfo->offset;
   } else {
      length = (size_t)pMemoryMapInfo->size;
   }

   WRAPPER_LOG(info, "Mapping memory %p, address %p offset %llu size %zu",
               (void *)(uintptr_t)pMemoryMapInfo->memory, placed_info->pPlacedAddress,
               (unsigned long long)pMemoryMapInfo->offset, length);

   void *addr = mmap(placed_info->pPlacedAddress, length, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_FIXED, fd, (off_t)pMemoryMapInfo->offset);
   if (addr == MAP_FAILED) {
      WRAPPER_LOG(error, "mmap failed: error %d", errno);
      goto fail;
   }

   mem->map_address = addr;
   mem->map_size = length;

out:
   simple_mtx_unlock(&device->resource_mutex);
   *ppData = mem->map_address;   /* VK_EXT_map_memory_placed: *ppData == pPlacedAddress */
   return VK_SUCCESS;

fail:
   simple_mtx_unlock(&device->resource_mutex);
   return vk_error(device, result);
}

VKAPI_ATTR void VKAPI_CALL
wrapper_UnmapMemory(VkDevice _device, VkDeviceMemory _memory)
{
   vk_common_UnmapMemory(_device, _memory);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_UnmapMemory2KHR(VkDevice _device,
                        const VkMemoryUnmapInfoKHR *pMemoryUnmapInfo)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   struct wrapper_device_memory *mem;
   VkResult result = VK_SUCCESS;

   mem = wrapper_device_memory_from_handle(device, pMemoryUnmapInfo->memory);

   simple_mtx_lock(&device->resource_mutex);

   if (!mem || !mem->map_address) {
      /* not one of our placed mappings: the driver owns this map */
      simple_mtx_unlock(&device->resource_mutex);
      device->dispatch_table.UnmapMemory(device->dispatch_handle,
         pMemoryUnmapInfo->memory);
      return VK_SUCCESS;
   }

   WRAPPER_LOG(info, "Emulating vkUnmapMemory2KHR");

   if (pMemoryUnmapInfo->flags & VK_MEMORY_UNMAP_RESERVE_BIT_EXT) {
      void *res = mmap(mem->map_address, mem->map_size, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
      if (res == MAP_FAILED) {
         WRAPPER_LOG(error, "Failed to replace mapping with reserved memory");
         result = VK_ERROR_MEMORY_MAP_FAILED;
      }
   } else {
      munmap(mem->map_address, mem->map_size);
   }

   mem->map_size = 0;
   mem->map_address = NULL;

   simple_mtx_unlock(&device->resource_mutex);
   return result == VK_SUCCESS ? VK_SUCCESS : vk_error(device, result);
}
