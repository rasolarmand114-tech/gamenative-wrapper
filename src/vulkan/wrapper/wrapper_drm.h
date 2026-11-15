#ifndef WRAPPER_DRM_H
#define WRAPPER_DRM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * DRM node + libdrm access for the wrapper.
 *
 * libdrm is loaded from the vendor partition (/vendor/lib64/libdrm.so) when the
 * linker namespace lets us, otherwise every operation falls back to plain DRM
 * ioctls, so nothing here ever depends on libdrm being present. A render node
 * (/dev/dri/renderD*) is preferred over a primary node (/dev/dri/card*).
 */
struct wrapper_drm {
   int   fd;                    /* opened DRM node, -1 when none        */
   char  node_path[64];
   bool  is_render_node;
   bool  prime_import;          /* DRM_PRIME_CAP_IMPORT                 */
   bool  prime_export;          /* DRM_PRIME_CAP_EXPORT                 */
   bool  dumb_buffer;           /* DRM_CAP_DUMB_BUFFER                  */
   char  driver_name[32];
   char  libdrm_path[64];       /* empty when libdrm could not be loaded */
   void *libdrm;

   /* libdrm entry points, NULL when missing */
   int   (*fn_get_devices2)(uint32_t flags, void **devices, int max);
   void  (*fn_free_devices)(void **devices, int count);
   void *(*fn_get_version)(int fd);
   void  (*fn_free_version)(void *version);
   int   (*fn_get_cap)(int fd, uint64_t cap, uint64_t *value);
   int   (*fn_prime_handle_to_fd)(int fd, uint32_t handle, uint32_t flags, int *prime_fd);
   int   (*fn_prime_fd_to_handle)(int fd, int prime_fd, uint32_t *handle);
   int   (*fn_close_buffer_handle)(int fd, uint32_t handle);
};

/* Never fails: on error wrapper_drm.fd stays -1 and the feature flags false. */
void wrapper_drm_init(struct wrapper_drm *drm);
void wrapper_drm_finish(struct wrapper_drm *drm);

/* Allocates `size` bytes of CPU-mappable, GPU-importable memory from the DRM
 * node and returns it as a dma-buf fd (the caller owns it), or -1. */
int wrapper_drm_alloc_dmabuf(struct wrapper_drm *drm, size_t size);

#endif
