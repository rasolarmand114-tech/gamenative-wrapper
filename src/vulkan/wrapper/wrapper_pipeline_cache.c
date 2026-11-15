/*
 * Persistent pipeline cache.
 *
 * Mali's shader compiler is slow and Android gives apps no system-wide
 * pipeline cache, so every launch recompiled every DXVK/vkd3d/LSFG pipeline.
 * The wrapper now owns one VkPipelineCache per device that is
 *   - seeded from /sdcard/wrapper/pipeline_cache/<vendor>_<device>_<uuid>.bin
 *     (only if its header matches this exact driver build),
 *   - used for every vkCreate*Pipelines call the application makes with
 *     VK_NULL_HANDLE (caches the application supplies are left alone),
 *   - written back atomically (tmp file + rename) every WRAPPER_PCACHE_SAVE_INTERVAL_SEC
 *     seconds if something new was compiled, and once more at device teardown.
 *
 * WRAPPER_PIPELINE_CACHE=0 disables it, WRAPPER_PIPELINE_CACHE_DIR moves it.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "wrapper_private.h"
#include "wrapper_log.h"
#include "wrapper_entrypoints.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define WRAPPER_PCACHE_DIR                 "/sdcard/wrapper/pipeline_cache"
#define WRAPPER_PCACHE_SAVE_INTERVAL_SEC   45
#define WRAPPER_PCACHE_MAX_BYTES           (512ull * 1024ull * 1024ull)
/* VkPipelineCacheHeaderVersionOne: 4 x uint32 + pipelineCacheUUID */
#define WRAPPER_PCACHE_HEADER_BYTES        (16u + VK_UUID_SIZE)

struct wrapper_pcache {
   VkPipelineCache cache;
   char path[512];

   pthread_t thread;
   bool thread_running;
   bool stop;
   pthread_mutex_t lock;       /* protects `stop` + the timed wait */
   pthread_cond_t cond;

   uint32_t created_since_save;   /* touched atomically */
   size_t saved_size;
};

static bool
pcache_enabled(void)
{
   const char *env = getenv("WRAPPER_PIPELINE_CACHE");

   return !env || atoi(env) != 0;
}

static void
pcache_mkdir_p(const char *dir)
{
   char tmp[512];

   snprintf(tmp, sizeof(tmp), "%s", dir);
   for (char *p = tmp + 1; *p; p++) {
      if (*p != '/')
         continue;
      *p = '\0';
      mkdir(tmp, 0770);
      *p = '/';
   }
   mkdir(tmp, 0770);
}

static uint8_t *
pcache_read_file(const char *path, size_t *out_size)
{
   FILE *f = fopen(path, "rb");
   uint8_t *data = NULL;
   long size;

   *out_size = 0;
   if (!f)
      return NULL;

   if (fseek(f, 0, SEEK_END) != 0 || (size = ftell(f)) <= 0 ||
       (unsigned long long)size > WRAPPER_PCACHE_MAX_BYTES ||
       fseek(f, 0, SEEK_SET) != 0) {
      fclose(f);
      return NULL;
   }

   data = malloc((size_t)size);
   if (data && fread(data, 1, (size_t)size, f) == (size_t)size) {
      *out_size = (size_t)size;
   } else {
      free(data);
      data = NULL;
   }
   fclose(f);
   return data;
}

static bool
pcache_header_matches(const uint8_t *data, size_t size,
                      const VkPhysicalDeviceProperties *props)
{
   uint32_t header_size, header_version, vendor_id, device_id;

   if (size < WRAPPER_PCACHE_HEADER_BYTES)
      return false;

   memcpy(&header_size, data, 4);
   memcpy(&header_version, data + 4, 4);
   memcpy(&vendor_id, data + 8, 4);
   memcpy(&device_id, data + 12, 4);

   return header_size >= WRAPPER_PCACHE_HEADER_BYTES && header_size <= size &&
          header_version == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
          vendor_id == props->vendorID && device_id == props->deviceID &&
          memcmp(data + 16, props->pipelineCacheUUID, VK_UUID_SIZE) == 0;
}

static void
pcache_save(struct wrapper_device *device, struct wrapper_pcache *pc)
{
   const struct vk_device_dispatch_table *dt = &device->dispatch_table;
   char tmp_path[sizeof(pc->path) + 32];
   size_t size = 0, done = 0;
   uint8_t *data;
   int fd;

   if (dt->GetPipelineCacheData(device->dispatch_handle, pc->cache, &size, NULL) != VK_SUCCESS ||
       size <= WRAPPER_PCACHE_HEADER_BYTES || size > WRAPPER_PCACHE_MAX_BYTES)
      return;

   data = malloc(size);
   if (!data)
      return;

   /* VK_INCOMPLETE means the cache grew between the two calls: not a valid blob */
   if (dt->GetPipelineCacheData(device->dispatch_handle, pc->cache, &size, data) != VK_SUCCESS) {
      free(data);
      return;
   }

   snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%d", pc->path, (int)getpid());
   fd = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0660);
   if (fd < 0) {
      WRAPPER_LOG(error, "pipeline cache: cannot write %s (errno %d)", tmp_path, errno);
      free(data);
      return;
   }

   while (done < size) {
      ssize_t n = write(fd, data + done, size - done);

      if (n < 0 && errno == EINTR)
         continue;
      if (n <= 0)
         break;
      done += (size_t)n;
   }
   close(fd);
   free(data);

   if (done != size || rename(tmp_path, pc->path) != 0) {
      unlink(tmp_path);
      WRAPPER_LOG(error, "pipeline cache: saving %s failed", pc->path);
      return;
   }

   pc->saved_size = size;
   WRAPPER_LOG(info, "pipeline cache: saved %zu bytes to %s", size, pc->path);
}

static void *
pcache_thread(void *arg)
{
   struct wrapper_device *device = arg;
   struct wrapper_pcache *pc = device->pcache;

   pthread_mutex_lock(&pc->lock);
   while (!pc->stop) {
      struct timespec deadline;

      clock_gettime(CLOCK_MONOTONIC, &deadline);
      deadline.tv_sec += WRAPPER_PCACHE_SAVE_INTERVAL_SEC;
      pthread_cond_timedwait(&pc->cond, &pc->lock, &deadline);

      if (pc->stop)
         break;
      if (__atomic_exchange_n(&pc->created_since_save, 0, __ATOMIC_RELAXED) == 0)
         continue;

      pthread_mutex_unlock(&pc->lock);
      pcache_save(device, pc);
      pthread_mutex_lock(&pc->lock);
   }
   pthread_mutex_unlock(&pc->lock);
   return NULL;
}

struct wrapper_pcache *
wrapper_pcache_create(struct wrapper_device *device)
{
   const VkPhysicalDeviceProperties *props = &device->physical->properties2.properties;
   const struct vk_device_dispatch_table *dt = &device->dispatch_table;
   struct wrapper_pcache *pc;
   const char *dir;
   char uuid[2 * VK_UUID_SIZE + 1];
   uint8_t *seed = NULL;
   size_t seed_size = 0;
   VkResult result;
   pthread_condattr_t attr;

   if (!pcache_enabled() || !dt->CreatePipelineCache || !dt->GetPipelineCacheData ||
       !dt->DestroyPipelineCache)
      return NULL;

   pc = calloc(1, sizeof(*pc));
   if (!pc)
      return NULL;

   dir = getenv("WRAPPER_PIPELINE_CACHE_DIR");
   if (!dir || !dir[0])
      dir = WRAPPER_PCACHE_DIR;

   for (unsigned i = 0; i < VK_UUID_SIZE; i++)
      snprintf(uuid + 2 * i, 3, "%02x", props->pipelineCacheUUID[i]);
   snprintf(pc->path, sizeof(pc->path), "%s/pc_%04x_%08x_%s.bin",
            dir, props->vendorID, props->deviceID, uuid);

   pcache_mkdir_p(dir);

   seed = pcache_read_file(pc->path, &seed_size);
   if (seed && !pcache_header_matches(seed, seed_size, props)) {
      WRAPPER_LOG(info, "pipeline cache: %s belongs to another driver build, ignoring", pc->path);
      free(seed);
      seed = NULL;
      seed_size = 0;
   }

   VkPipelineCacheCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
      .initialDataSize = seed_size,
      .pInitialData = seed,
   };
   result = dt->CreatePipelineCache(device->dispatch_handle, &create_info, NULL, &pc->cache);
   if (result != VK_SUCCESS && seed) {
      /* a corrupt/unusable seed must never keep the cache from working */
      create_info.initialDataSize = 0;
      create_info.pInitialData = NULL;
      seed_size = 0;
      result = dt->CreatePipelineCache(device->dispatch_handle, &create_info, NULL, &pc->cache);
   }
   free(seed);

   if (result != VK_SUCCESS) {
      WRAPPER_LOG(error, "pipeline cache: vkCreatePipelineCache failed, res %d", result);
      free(pc);
      return NULL;
   }

   pc->saved_size = seed_size;
   WRAPPER_LOG(info, "pipeline cache: %s (%zu bytes loaded)", pc->path, seed_size);

   pthread_condattr_init(&attr);
   pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
   pthread_cond_init(&pc->cond, &attr);
   pthread_condattr_destroy(&attr);
   pthread_mutex_init(&pc->lock, NULL);

   device->pcache = pc;
   if (pthread_create(&pc->thread, NULL, pcache_thread, device) == 0)
      pc->thread_running = true;   /* otherwise: save at teardown only */

   return pc;
}

void
wrapper_pcache_destroy(struct wrapper_device *device)
{
   struct wrapper_pcache *pc = device->pcache;

   if (!pc)
      return;

   if (pc->thread_running) {
      pthread_mutex_lock(&pc->lock);
      pc->stop = true;
      pthread_cond_signal(&pc->cond);
      pthread_mutex_unlock(&pc->lock);
      pthread_join(pc->thread, NULL);
   }

   if (__atomic_exchange_n(&pc->created_since_save, 0, __ATOMIC_RELAXED) != 0)
      pcache_save(device, pc);

   device->dispatch_table.DestroyPipelineCache(device->dispatch_handle, pc->cache, NULL);

   pthread_cond_destroy(&pc->cond);
   pthread_mutex_destroy(&pc->lock);
   free(pc);
   device->pcache = NULL;
}

VkPipelineCache
wrapper_pcache_handle(struct wrapper_device *device)
{
   return device->pcache ? device->pcache->cache : VK_NULL_HANDLE;
}

void
wrapper_pcache_note_created(struct wrapper_device *device, uint32_t count)
{
   if (device->pcache && count)
      __atomic_fetch_add(&device->pcache->created_since_save, count, __ATOMIC_RELAXED);
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateGraphicsPipelines(VkDevice _device, VkPipelineCache pipelineCache,
                                uint32_t createInfoCount,
                                const VkGraphicsPipelineCreateInfo *pCreateInfos,
                                const VkAllocationCallbacks *pAllocator,
                                VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkPipelineCache own = pipelineCache == VK_NULL_HANDLE ?
                         wrapper_pcache_handle(device) : VK_NULL_HANDLE;
   VkResult result;

   result = device->dispatch_table.CreateGraphicsPipelines(device->dispatch_handle,
      own != VK_NULL_HANDLE ? own : pipelineCache, createInfoCount, pCreateInfos,
      pAllocator, pPipelines);

   if (result == VK_SUCCESS && own != VK_NULL_HANDLE)
      wrapper_pcache_note_created(device, createInfoCount);

   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
wrapper_CreateComputePipelines(VkDevice _device, VkPipelineCache pipelineCache,
                               uint32_t createInfoCount,
                               const VkComputePipelineCreateInfo *pCreateInfos,
                               const VkAllocationCallbacks *pAllocator,
                               VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(wrapper_device, device, _device);
   VkPipelineCache own = pipelineCache == VK_NULL_HANDLE ?
                         wrapper_pcache_handle(device) : VK_NULL_HANDLE;
   VkResult result;

   result = device->dispatch_table.CreateComputePipelines(device->dispatch_handle,
      own != VK_NULL_HANDLE ? own : pipelineCache, createInfoCount, pCreateInfos,
      pAllocator, pPipelines);

   if (result == VK_SUCCESS && own != VK_NULL_HANDLE)
      wrapper_pcache_note_created(device, createInfoCount);

   return result;
}
