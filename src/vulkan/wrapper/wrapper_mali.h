#ifndef WRAPPER_MALI_H
#define WRAPPER_MALI_H

#include <stdbool.h>
#include <stdint.h>

/* PCI-style vendor id Arm's Vulkan driver reports for every Mali/Immortalis GPU. */
#define WRAPPER_MALI_VENDOR_ID 0x13B5u

/* kbase (Arm's kernel driver) character device. */
#define WRAPPER_KBASE_NODE "/dev/mali0"

struct wrapper_mali_info {
   /* ---- read straight from the kernel through /dev/mali0 (kbase) ---- */
   bool     kbase_ok;               /* handshake + GPU props succeeded      */
   uint32_t kbase_uk_major;         /* kbase user/kernel ABI version        */
   uint32_t kbase_uk_minor;
   uint64_t shader_present;         /* shader-core bitmask                  */
   uint32_t l2_line_bytes;
   uint32_t l2_cache_bytes;
   uint32_t l2_slices;
   uint32_t max_threads;
   uint32_t max_workgroup_size;
   uint32_t max_registers;
   uint32_t freq_khz_max;

   /* ---- merged view (kbase first, Vulkan data as fallback) ---- */
   uint32_t gpu_id;                 /* raw GPU_ID register (0 = unknown)    */
   uint32_t product_id;             /* gpu_id >> 16                         */
   uint32_t arch_major;             /* gpu_id >> 28: 6/7 Bifrost, 9+ Valhall */
   uint32_t core_count;             /* shader cores (0 = unknown)           */
   bool     core_count_from_kbase;
   bool     is_valhall;             /* arch >= 9                            */
   bool     is_csf;                 /* arch >= 10 (command-stream frontend) */
};

/* Opens /dev/mali0, performs the kbase handshake, creates a throw-away kbase
 * context and reads the GPU properties. The file descriptor is closed again
 * before returning. Never fatal: returns false (and leaves *info zeroed) if the
 * node is missing, not accessible (SELinux) or the ABI is not understood. */
bool wrapper_kbase_probe(struct wrapper_mali_info *info);

/* Fills the merged fields. `vk_device_id` is the deviceID the Vulkan driver
 * reports (the raw GPU_ID on Arm's driver); `vk_device_name` is used only to
 * recover the shader-core count ("MC10" / "MP2") when kbase is unavailable. */
void wrapper_mali_finish_info(struct wrapper_mali_info *info,
                              const char *vk_device_name,
                              uint32_t vk_device_id);

#endif
