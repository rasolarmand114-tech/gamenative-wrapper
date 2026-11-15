/*
 * Mali identification through kbase (/dev/mali0).
 *
 * The wrapper used to guess the GPU generation and shader-core count from the
 * marketing name string. kbase knows the real values (raw GPU_ID, shader
 * present mask, L2 geometry, thread limits), so ask it directly. Everything
 * here is best effort and read-only: a failure just means "fall back to what
 * the Vulkan driver says".
 *
 * The UAPI structures/ioctl numbers below are the stable part of Arm's kbase
 * ABI (mali_kbase_ioctl.h) that every Mali kernel driver from Bifrost onwards
 * has exported: VERSION_CHECK, SET_FLAGS and GET_GPUPROPS.
 */

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "wrapper_log.h"
#include "wrapper_mali.h"

#define KBASE_IOCTL_TYPE 0x80

struct kbase_ioctl_version_check {
   uint16_t major;
   uint16_t minor;
};
#define KBASE_IOCTL_VERSION_CHECK \
   _IOWR(KBASE_IOCTL_TYPE, 0, struct kbase_ioctl_version_check)

struct kbase_ioctl_set_flags {
   uint32_t create_flags;
};
#define KBASE_IOCTL_SET_FLAGS \
   _IOW(KBASE_IOCTL_TYPE, 1, struct kbase_ioctl_set_flags)

struct kbase_ioctl_get_gpuprops {
   uint64_t buffer;
   uint32_t size;
   uint32_t flags;
};
#define KBASE_IOCTL_GET_GPUPROPS \
   _IOW(KBASE_IOCTL_TYPE, 3, struct kbase_ioctl_get_gpuprops)

/* GPU property ids (key = id << 2 | size_code, size_code 0..3 = 1/2/4/8 bytes) */
#define KBASE_GPUPROP_PRODUCT_ID           1
#define KBASE_GPUPROP_GPU_FREQ_KHZ_MAX     6
#define KBASE_GPUPROP_L2_LOG2_LINE_SIZE    13
#define KBASE_GPUPROP_L2_LOG2_CACHE_SIZE   14
#define KBASE_GPUPROP_L2_NUM_L2_SLICES     15
#define KBASE_GPUPROP_MAX_THREADS          18
#define KBASE_GPUPROP_MAX_WORKGROUP_SIZE   19
#define KBASE_GPUPROP_MAX_REGISTERS        21
#define KBASE_GPUPROP_RAW_SHADER_PRESENT   25
#define KBASE_GPUPROP_RAW_GPU_ID           55
#define KBASE_GPUPROP_MAX_ID               128

#define KBASE_PROPS_MAX_BYTES (64u * 1024u)

struct kbase_props {
   uint64_t value[KBASE_GPUPROP_MAX_ID];
   uint8_t  have[KBASE_GPUPROP_MAX_ID];
};

static int
kbase_ioctl(int fd, unsigned long request, void *arg)
{
   int ret;

   do {
      ret = ioctl(fd, request, arg);
   } while (ret == -1 && errno == EINTR);

   return ret;
}

static void
kbase_parse_props(const uint8_t *buf, size_t len, struct kbase_props *out)
{
   size_t off = 0;

   while (off + 4 <= len) {
      uint32_t key = (uint32_t)buf[off] | ((uint32_t)buf[off + 1] << 8) |
                     ((uint32_t)buf[off + 2] << 16) | ((uint32_t)buf[off + 3] << 24);
      uint32_t id = key >> 2;
      size_t bytes = (size_t)1 << (key & 3);
      uint64_t v = 0;

      off += 4;
      if (off + bytes > len)
         break;

      for (size_t i = 0; i < bytes; i++)
         v |= (uint64_t)buf[off + i] << (8 * i);
      off += bytes;

      if (id < KBASE_GPUPROP_MAX_ID) {
         out->value[id] = v;
         out->have[id] = 1;
      }
   }
}

static unsigned
popcount64(uint64_t v)
{
   unsigned n = 0;

   while (v) {
      v &= v - 1;
      n++;
   }
   return n;
}

bool
wrapper_kbase_probe(struct wrapper_mali_info *info)
{
   struct kbase_ioctl_version_check ver = { 0, 0 };
   struct kbase_ioctl_set_flags flags = { 0 };
   struct kbase_ioctl_get_gpuprops gp = { 0 };
   struct kbase_props *props = NULL;
   uint8_t *buf = NULL;
   bool ok = false;
   int fd, size;

   memset(info, 0, sizeof(*info));

   fd = open(WRAPPER_KBASE_NODE, O_RDWR | O_CLOEXEC);
   if (fd < 0) {
      WRAPPER_LOG(info, "kbase: cannot open %s (errno %d)", WRAPPER_KBASE_NODE, errno);
      return false;
   }

   /* Asking for major 0 makes kbase answer with the ABI version it speaks and
    * complete the handshake with exactly that version. */
   if (kbase_ioctl(fd, KBASE_IOCTL_VERSION_CHECK, &ver) < 0) {
      WRAPPER_LOG(info, "kbase: version handshake failed (errno %d)", errno);
      goto out;
   }

   if (kbase_ioctl(fd, KBASE_IOCTL_SET_FLAGS, &flags) < 0) {
      WRAPPER_LOG(info, "kbase: context setup failed (errno %d)", errno);
      goto out;
   }

   gp.buffer = 0;
   gp.size = 0;
   gp.flags = 0;
   size = kbase_ioctl(fd, KBASE_IOCTL_GET_GPUPROPS, &gp);
   if (size <= 0 || (unsigned)size > KBASE_PROPS_MAX_BYTES) {
      WRAPPER_LOG(info, "kbase: unexpected GPU property size %d", size);
      goto out;
   }

   buf = calloc(1, (size_t)size);
   props = calloc(1, sizeof(*props));
   if (!buf || !props)
      goto out;

   gp.buffer = (uint64_t)(uintptr_t)buf;
   gp.size = (uint32_t)size;
   gp.flags = 0;
   if (kbase_ioctl(fd, KBASE_IOCTL_GET_GPUPROPS, &gp) < 0) {
      WRAPPER_LOG(info, "kbase: reading GPU properties failed (errno %d)", errno);
      goto out;
   }

   kbase_parse_props(buf, (size_t)size, props);

#define HAVE(id) (props->have[id])
#define VAL(id)  (props->value[id])

   if (HAVE(KBASE_GPUPROP_RAW_GPU_ID))
      info->gpu_id = (uint32_t)VAL(KBASE_GPUPROP_RAW_GPU_ID);
   else if (HAVE(KBASE_GPUPROP_PRODUCT_ID))
      info->gpu_id = (uint32_t)VAL(KBASE_GPUPROP_PRODUCT_ID) << 16;

   if (HAVE(KBASE_GPUPROP_RAW_SHADER_PRESENT)) {
      info->shader_present = VAL(KBASE_GPUPROP_RAW_SHADER_PRESENT);
      info->core_count = popcount64(info->shader_present);
      info->core_count_from_kbase = info->core_count > 0;
   }

   if (HAVE(KBASE_GPUPROP_L2_LOG2_LINE_SIZE) &&
       VAL(KBASE_GPUPROP_L2_LOG2_LINE_SIZE) < 16)
      info->l2_line_bytes = 1u << VAL(KBASE_GPUPROP_L2_LOG2_LINE_SIZE);
   if (HAVE(KBASE_GPUPROP_L2_LOG2_CACHE_SIZE) &&
       VAL(KBASE_GPUPROP_L2_LOG2_CACHE_SIZE) < 31)
      info->l2_cache_bytes = 1u << VAL(KBASE_GPUPROP_L2_LOG2_CACHE_SIZE);
   if (HAVE(KBASE_GPUPROP_L2_NUM_L2_SLICES))
      info->l2_slices = (uint32_t)VAL(KBASE_GPUPROP_L2_NUM_L2_SLICES);
   if (HAVE(KBASE_GPUPROP_MAX_THREADS))
      info->max_threads = (uint32_t)VAL(KBASE_GPUPROP_MAX_THREADS);
   if (HAVE(KBASE_GPUPROP_MAX_WORKGROUP_SIZE))
      info->max_workgroup_size = (uint32_t)VAL(KBASE_GPUPROP_MAX_WORKGROUP_SIZE);
   if (HAVE(KBASE_GPUPROP_MAX_REGISTERS))
      info->max_registers = (uint32_t)VAL(KBASE_GPUPROP_MAX_REGISTERS);
   if (HAVE(KBASE_GPUPROP_GPU_FREQ_KHZ_MAX))
      info->freq_khz_max = (uint32_t)VAL(KBASE_GPUPROP_GPU_FREQ_KHZ_MAX);

#undef HAVE
#undef VAL

   info->kbase_uk_major = ver.major;
   info->kbase_uk_minor = ver.minor;
   info->kbase_ok = info->gpu_id != 0;
   ok = info->kbase_ok;

   WRAPPER_LOG(info,
      "kbase: uk %u.%u GPU_ID 0x%08x shader cores %u (mask 0x%llx) L2 %u KiB/%u B x%u "
      "threads %u workgroup %u regs %u maxfreq %u kHz",
      info->kbase_uk_major, info->kbase_uk_minor, info->gpu_id, info->core_count,
      (unsigned long long)info->shader_present, info->l2_cache_bytes / 1024,
      info->l2_line_bytes, info->l2_slices, info->max_threads,
      info->max_workgroup_size, info->max_registers, info->freq_khz_max);

out:
   free(buf);
   free(props);
   close(fd);
   if (!ok)
      memset(info, 0, sizeof(*info));
   return ok;
}

/* "Mali-G710 MC10" / "Mali-G52 MP2" -> 10 / 2, or 0 when there is no count. */
static uint32_t
core_count_from_name(const char *name)
{
   const char *suffix;
   int count;

   if (!name)
      return 0;

   suffix = strstr(name, "MC");
   if (!suffix)
      suffix = strstr(name, "MP");
   if (!suffix)
      return 0;

   count = atoi(suffix + 2);
   return count > 0 ? (uint32_t)count : 0;
}

void
wrapper_mali_finish_info(struct wrapper_mali_info *info,
                         const char *vk_device_name,
                         uint32_t vk_device_id)
{
   /* On Arm's Vulkan driver deviceID carries the raw GPU_ID register. */
   if (!info->gpu_id)
      info->gpu_id = vk_device_id;

   info->product_id = info->gpu_id >> 16;
   info->arch_major = info->gpu_id >> 28;
   info->is_valhall = info->arch_major >= 9;
   info->is_csf = info->arch_major >= 10;

   if (!info->core_count)
      info->core_count = core_count_from_name(vk_device_name);
}
