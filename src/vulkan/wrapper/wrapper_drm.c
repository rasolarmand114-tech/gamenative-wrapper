#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include "wrapper_drm.h"
#include "wrapper_log.h"

#ifdef __LP64__
#define WRAPPER_VENDOR_LIBDRM "/vendor/lib64/libdrm.so"
#else
#define WRAPPER_VENDOR_LIBDRM "/vendor/lib/libdrm.so"
#endif

/* ---- minimal copy of the DRM uapi we need (stable kernel ABI) ---- */
#define WDRM_NODE_PRIMARY 0
#define WDRM_NODE_RENDER  2

#define WDRM_CAP_DUMB_BUFFER   0x1
#define WDRM_CAP_PRIME         0x5
#define WDRM_PRIME_CAP_IMPORT  0x1
#define WDRM_PRIME_CAP_EXPORT  0x2

struct wdrm_get_cap {
   uint64_t capability;
   uint64_t value;
};
struct wdrm_create_dumb {
   uint32_t height, width, bpp, flags, handle, pitch;
   uint64_t size;
};
struct wdrm_gem_close {
   uint32_t handle;
   uint32_t pad;
};
struct wdrm_prime_handle {
   uint32_t handle;
   uint32_t flags;
   int32_t  fd;
};

#define WDRM_IOCTL_GET_CAP            _IOWR('d', 0x0c, struct wdrm_get_cap)
#define WDRM_IOCTL_GEM_CLOSE          _IOW('d', 0x09, struct wdrm_gem_close)
#define WDRM_IOCTL_PRIME_HANDLE_TO_FD _IOWR('d', 0x2d, struct wdrm_prime_handle)
#define WDRM_IOCTL_MODE_CREATE_DUMB   _IOWR('d', 0xB2, struct wdrm_create_dumb)

/* Leading part of libdrm's drmDevice / drmVersion (only what we read). */
struct wdrm_device_head {
   char **nodes;
   int available_nodes;
};
struct wdrm_version_head {
   int version_major;
   int version_minor;
   int version_patchlevel;
   int name_len;
   char *name;
};

static int
wdrm_ioctl(int fd, unsigned long request, void *arg)
{
   int ret;

   do {
      ret = ioctl(fd, request, arg);
   } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

   return ret;
}

static void
wdrm_load_libdrm(struct wrapper_drm *drm)
{
   static const char *const candidates[] = {
      WRAPPER_VENDOR_LIBDRM,
      "libdrm.so",
   };

   for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
      drm->libdrm = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
      if (drm->libdrm) {
         snprintf(drm->libdrm_path, sizeof(drm->libdrm_path), "%s", candidates[i]);
         break;
      }
   }

   if (!drm->libdrm) {
      WRAPPER_LOG(info, "drm: libdrm not loadable (%s), using raw DRM ioctls", WRAPPER_VENDOR_LIBDRM);
      return;
   }

#define LOAD(field, sym) *(void **)&drm->field = dlsym(drm->libdrm, sym)
   LOAD(fn_get_devices2, "drmGetDevices2");
   LOAD(fn_free_devices, "drmFreeDevices");
   LOAD(fn_get_version, "drmGetVersion");
   LOAD(fn_free_version, "drmFreeVersion");
   LOAD(fn_get_cap, "drmGetCap");
   LOAD(fn_prime_handle_to_fd, "drmPrimeHandleToFD");
   LOAD(fn_prime_fd_to_handle, "drmPrimeFDToHandle");
   LOAD(fn_close_buffer_handle, "drmCloseBufferHandle");
#undef LOAD

   WRAPPER_LOG(info, "drm: loaded %s", drm->libdrm_path);
}

static int
wdrm_open_node(struct wrapper_drm *drm, const char *path, bool render)
{
   int fd = open(path, O_RDWR | O_CLOEXEC);

   if (fd < 0)
      return -1;

   drm->fd = fd;
   drm->is_render_node = render;
   snprintf(drm->node_path, sizeof(drm->node_path), "%s", path);
   return 0;
}

/* Preferred path: ask libdrm which nodes exist. */
static void
wdrm_find_node_with_libdrm(struct wrapper_drm *drm)
{
   void *devices[16] = { 0 };
   int count;

   if (!drm->fn_get_devices2)
      return;

   count = drm->fn_get_devices2(0, devices, 16);
   if (count <= 0)
      return;

   /* render nodes first, then primary nodes */
   for (int pass = 0; pass < 2 && drm->fd < 0; pass++) {
      int type = pass == 0 ? WDRM_NODE_RENDER : WDRM_NODE_PRIMARY;

      for (int i = 0; i < count && drm->fd < 0; i++) {
         struct wdrm_device_head *dev = devices[i];

         if (!dev || !dev->nodes || !(dev->available_nodes & (1 << type)) ||
             !dev->nodes[type])
            continue;
         wdrm_open_node(drm, dev->nodes[type], type == WDRM_NODE_RENDER);
      }
   }

   if (drm->fn_free_devices)
      drm->fn_free_devices(devices, count);
}

/* Fallback path: look into /dev/dri ourselves. */
static void
wdrm_find_node_by_scan(struct wrapper_drm *drm)
{
   static const struct { const char *prefix; bool render; } kinds[] = {
      { "renderD", true },
      { "card", false },
   };

   for (int k = 0; k < 2 && drm->fd < 0; k++) {
      DIR *dir = opendir("/dev/dri");
      struct dirent *ent;

      if (!dir)
         return;

      while ((ent = readdir(dir)) != NULL && drm->fd < 0) {
         char path[64];

         if (strncmp(ent->d_name, kinds[k].prefix, strlen(kinds[k].prefix)) != 0)
            continue;
         snprintf(path, sizeof(path), "/dev/dri/%s", ent->d_name);
         wdrm_open_node(drm, path, kinds[k].render);
      }
      closedir(dir);
   }
}

static void
wdrm_query_driver_name(struct wrapper_drm *drm)
{
   if (drm->fn_get_version) {
      struct wdrm_version_head *ver = drm->fn_get_version(drm->fd);

      if (ver) {
         if (ver->name)
            snprintf(drm->driver_name, sizeof(drm->driver_name), "%s", ver->name);
         if (drm->fn_free_version)
            drm->fn_free_version(ver);
      }
   }

   if (!drm->driver_name[0]) {
      /* /sys/dev/char/<maj>:<min>/device/driver -> .../<driver> */
      struct stat st;
      char link[96], target[128];
      ssize_t n;

      if (fstat(drm->fd, &st) != 0)
         return;
      snprintf(link, sizeof(link), "/sys/dev/char/%u:%u/device/driver",
               major(st.st_rdev), minor(st.st_rdev));
      n = readlink(link, target, sizeof(target) - 1);
      if (n > 0) {
         const char *base;

         target[n] = '\0';
         base = strrchr(target, '/');
         snprintf(drm->driver_name, sizeof(drm->driver_name), "%s", base ? base + 1 : target);
      }
   }
}

static bool
wdrm_get_cap(struct wrapper_drm *drm, uint64_t cap, uint64_t *value)
{
   struct wdrm_get_cap args = { .capability = cap };

   *value = 0;
   if (drm->fn_get_cap)
      return drm->fn_get_cap(drm->fd, cap, value) == 0;

   if (wdrm_ioctl(drm->fd, WDRM_IOCTL_GET_CAP, &args) != 0)
      return false;
   *value = args.value;
   return true;
}

void
wrapper_drm_init(struct wrapper_drm *drm)
{
   const char *forced = getenv("WRAPPER_DRM_NODE");
   uint64_t value;

   memset(drm, 0, sizeof(*drm));
   drm->fd = -1;

   wdrm_load_libdrm(drm);

   if (forced && forced[0])
      wdrm_open_node(drm, forced, strstr(forced, "renderD") != NULL);
   if (drm->fd < 0)
      wdrm_find_node_with_libdrm(drm);
   if (drm->fd < 0)
      wdrm_find_node_by_scan(drm);

   if (drm->fd < 0) {
      WRAPPER_LOG(info, "drm: no accessible DRM node");
      return;
   }

   wdrm_query_driver_name(drm);

   if (wdrm_get_cap(drm, WDRM_CAP_PRIME, &value)) {
      drm->prime_import = (value & WDRM_PRIME_CAP_IMPORT) != 0;
      drm->prime_export = (value & WDRM_PRIME_CAP_EXPORT) != 0;
   }
   if (wdrm_get_cap(drm, WDRM_CAP_DUMB_BUFFER, &value))
      drm->dumb_buffer = value != 0;

   WRAPPER_LOG(info, "drm: node %s (%s) driver '%s' prime import=%d export=%d dumb=%d",
               drm->node_path, drm->is_render_node ? "render" : "primary",
               drm->driver_name, drm->prime_import, drm->prime_export, drm->dumb_buffer);
}

void
wrapper_drm_finish(struct wrapper_drm *drm)
{
   if (drm->fd >= 0)
      close(drm->fd);
   if (drm->libdrm)
      dlclose(drm->libdrm);
   memset(drm, 0, sizeof(*drm));
   drm->fd = -1;
}

static void
wdrm_close_handle(struct wrapper_drm *drm, uint32_t handle)
{
   struct wdrm_gem_close args = { .handle = handle };

   if (drm->fn_close_buffer_handle) {
      drm->fn_close_buffer_handle(drm->fd, handle);
      return;
   }
   wdrm_ioctl(drm->fd, WDRM_IOCTL_GEM_CLOSE, &args);
}

int
wrapper_drm_alloc_dmabuf(struct wrapper_drm *drm, size_t size)
{
   struct wdrm_create_dumb create = { 0 };
   int prime_fd = -1;
   int ret;

   if (drm->fd < 0 || !drm->prime_export || !drm->dumb_buffer || size == 0)
      return -1;

   /* A dumb buffer is a width x height x bpp image; use 8 bpp and a wide row so
    * the byte size is just rows * 4096 and any alignment padding only adds. */
   create.width = 4096;
   create.bpp = 8;
   create.height = (uint32_t)((size + 4095) / 4096);
   if ((size_t)create.height * 4096 < size)
      return -1;

   if (wdrm_ioctl(drm->fd, WDRM_IOCTL_MODE_CREATE_DUMB, &create) != 0)
      return -1;

   if (drm->fn_prime_handle_to_fd) {
      ret = drm->fn_prime_handle_to_fd(drm->fd, create.handle,
                                       O_CLOEXEC | O_RDWR, &prime_fd);
   } else {
      struct wdrm_prime_handle prime = { .handle = create.handle,
                                         .flags = O_CLOEXEC | O_RDWR, .fd = -1 };

      ret = wdrm_ioctl(drm->fd, WDRM_IOCTL_PRIME_HANDLE_TO_FD, &prime);
      prime_fd = prime.fd;
   }

   /* the dma-buf keeps the buffer alive; the GEM handle is no longer needed */
   wdrm_close_handle(drm, create.handle);

   return ret == 0 ? prime_fd : -1;
}
