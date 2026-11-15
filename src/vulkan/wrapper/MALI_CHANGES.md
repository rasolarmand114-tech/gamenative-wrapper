# wrapper-25 (src/vulkan/wrapper) — نسخه‌ی Mali-only

> ⚠️ روی دستگاه واقعی و داخل درخت Mesa/NDK **کامپایل و تست نشده**. فقط این‌ها چک شدن:
> ماژول‌های جدید (`wrapper_mali.c`, `wrapper_drm.c`, `wrapper_pipeline_cache.c`, `wrapper_device_memory.c`) با
> هدرهای mock سینتکس‌چک شدن، parser مربوط به kbase و patcher مربوط به SPIR-V با تست واحد (و ASan) اجرا شدن،
> و بقیه فایل‌ها با مرور دستی + چک تعادل براکت و ارجاع‌های حذف‌شده.

## ۱) درخواست‌ها → چی انجام شد

| درخواست | وضعیت |
|---|---|
| حذف BCn | حذف کامل: `bcdec.h`, `wrapper_bcdec.*`, `wrapper_astc.h`, `wrapper_bcn_spv.h`, `wrapper_bcn_transcode.comp` + همه‌ی hookهای `CreateImage/ImageView/CmdCopyBufferToImage/BindBufferMemory/Fence` که فقط برای BCn بودن. `textureCompressionBC` و format-propertyهای BC **دقیقاً همونی که درایور Mali گزارش می‌ده** برمی‌گرده. |
| حذف adrenotools | حذف از `wrapper_instance.c` و `meson.build`. (پوشه‌ی `subprojects/libadrenotools` رو هم می‌تونی پاک کنی.) |
| حذف انتخاب نسخه‌ی درایور | `ADRENOTOOLS_*` حذف شد؛ فقط `dlopen("/system/lib64/libvulkan.so")` (روی ۳۲ بیت `/system/lib/…`). عمداً fallback به soname `libvulkan.so` نیست (داخل Wine/Termux ممکنه خودِ wrapper رو دوباره لود کنه → recursion). |
| فقط Mali | `enumerate_physical_device` هر GPU که vendorID ≠ 0x13B5 **و** driverID ≠ ARM_PROPRIETARY باشه رو لیست نمی‌کنه (اپ می‌بینه ۰ دستگاه). کدهای Adreno/Xclipse حذف شدن. |
| deviceID / vendorID واقعی | override های `WRAPPER_DEVICE_ID/VENDOR_ID/DEVICE_NAME/DRIVER_ID` و پیشوند `Wrapper(...)` و `driverName="Wrapper driver"` حذف شد. اسم، vendorID، deviceID، driverID/Name/Info همون چیزیه که درایور Mali می‌گه. فقط اگه deviceID صفر بود از GPU_ID کرنل پر می‌شه. |
| kbase (`/dev/mali0`) | `wrapper_mali.c`: handshake + ساخت context موقت + `GET_GPUPROPS` → GPU_ID خام، تعداد shader core، L2، thread/register limits. fd بعد از خوندن بسته می‌شه. جایگزین حدس زدن از روی اسم GPU شده (نسل Valhall/CSF، تعداد core برای شرط dynamic rendering). اگه باز نشه/SELinux ببنده → خودکار به اطلاعات Vulkan برمی‌گرده. |
| DRM node + libdrm | `wrapper_drm.c`: `dlopen("/vendor/lib64/libdrm.so")` (بعد `libdrm.so`)، پیدا کردن `renderD*` (بعد `card*`) با `drmGetDevices2` (یا اسکن `/dev/dri`)، caps (PRIME/dumb)، اسم درایور. بدون libdrm هم با ioctl خام کار می‌کنه. استفاده‌ی عملی: tier دوم allocator (بخش AHB) + گزارش در diag. |
| AHB کامل | `wrapper_device_memory.c` بازنویسی شد: backend حافظه‌ی host-visible **فقط AHardwareBuffer** (dma-heap / ion / opaque-fd حذف). dma-buf از DRM فقط وقتی استفاده می‌شه که resource صراحتاً AHB رو ممنوع کرده (handle type فقط dma-buf). اگه هیچ‌کدوم نشد، allocation به درایور forward می‌شه (به‌جای fail). |
| zero-copy | مسیر آپلود تکسچر دیگه CPU-transcode/staging نداره؛ حافظه‌ی AHB یک‌بار mmap می‌شه و GPU و CPU روی همون صفحه‌ها کار می‌کنن (`MAP_FIXED` روی آدرس placed). |
| بهینه‌سازی Mali | cache پایدار pipeline (پایین)، `QueueSubmit/QueueSubmit2` بدون `malloc` و بدون lock (قبلاً هر submit یه mutex + malloc داشت)، lookup حافظه با hash-table به‌جای لیست خطی، allocate بدون نگه‌داشتن mutex سراسری، SPIR-V patch بدون lock. |
| LSFG | حالت خاصی وجود نداره (LSFG لایه/برنامه‌ی جداست)؛ چیزایی که کمکش می‌کنه: pipeline cache (کامپایل compute shaderها رو در اجرای بعدی حذف می‌کنه)، submit/alloc سبک‌تر، external-memory/AHB پایدار. `WRAPPER_SUBGROUP_OPS=1` عملیات subgroup واقعی درایور رو نشون می‌ده. |

## ۲) Pipeline cache پایدار (جدید)
- یک `VkPipelineCache` برای هر device؛ برای هر `vkCreateGraphicsPipelines/ComputePipelines` که `VK_NULL_HANDLE` بده استفاده می‌شه (cache خود اپ دست‌نخورده می‌مونه).
- مسیر: `/sdcard/wrapper/pipeline_cache/pc_<vendor>_<device>_<uuid>.bin` (`WRAPPER_PIPELINE_CACHE_DIR` عوضش می‌کنه). فقط اگه header با همین build درایور بخونه بارگذاری می‌شه.
- ذخیره‌ی atomic (tmp + rename) هر ۴۵ ثانیه اگه چیز جدیدی کامپایل شده باشه + هنگام `DestroyDevice`.
- `WRAPPER_PIPELINE_CACHE=0` خاموشش می‌کنه. اگه مسیر قابل نوشتن نباشه بی‌صدا غیرفعاله.

## ۳) باگ‌های رفع‌شده
- `spirv_patcher.cpp`: `codeSize` (بایت) به‌عنوان تعداد word استفاده می‌شد → حلقه ۴× از انتهای بافر رد می‌شد (heap over-read).
- `CreateShaderModule`: کپی SPIR-V patch‌شده هیچ‌وقت free نمی‌شد (نشتی به‌ازای هر shader) + lock بی‌دلیل روی کل compile.
- `wrapper_BindBufferMemory`: unlock روی mutexی که lock نشده بود (حذف شد؛ ماجرا فقط برای BCn بود).
- `wrapper_FreeMemory`: `list_del` بدون lock (race)؛ `alloc_size` هیچ‌وقت set نمی‌شد؛ allocator نامعتبر (`&device->vk.alloc`) به درایور پاس می‌شد.
- `vkUnmapMemory` روی حافظه‌ی emulated که با map معمولی map شده بود، unmap درایور رو صدا نمی‌زد (map بعدی fail می‌شد).
- placed-map برای حافظه‌ای که emulation نداره، pointer اشتباه (غیر از آدرس خواسته‌شده) برمی‌گردوند → الان `VK_ERROR_MEMORY_MAP_FAILED`؛ offset طبق spec (`*ppData == pPlacedAddress`).
- fallback بعد از شکست emulation با `pNext` دست‌کاری‌شده‌ی اپ صدا زده می‌شد → الان chain اپ دقیقاً برمی‌گرده.
- `wrapper_CreateInstance`: حلقه‌ی extensionها تا `enabledExtensionCount` می‌رفت نه تا اندازه‌ی جدول (extensionهای غلط/تکراری به درایور می‌رفت).
- `wrapper_DestroyInstance`: wrapper-instance، physical deviceها، fd های DRM/heap هیچ‌وقت free/close نمی‌شدن.
- extensionهایی که wrapper «الزامی» می‌دونست حتی اگه درایور نداشت به `vkCreateDevice` داده می‌شد (→ `EXTENSION_NOT_PRESENT`)؛ الان فقط اگه درایور داشته باشه.
- شرط Valhall MC2+ دیگه کل device creation رو fail نمی‌کنه: هرچی درایور داره فعال می‌شه، بقیه فقط warning.
- لیست extensionهای درایور روی stack با اندازه‌ی ثابت بود؛ `VK_INCOMPLETE` بد هندل می‌شد.
- `WRAPPER_EXTENSION_BLACKLIST`: `strtok` روی خود env می‌نوشت و تطبیق substring بود (`VK_KHR_swapchain` اکستنشن‌های `…_mutable_format` رو هم می‌پروند).
- `maintenance5` فیک‌شده برای DXVK فقط روی Samsung feature-bit می‌گرفت؛ الان روی Mali هم.
- لاگ: `get_executable_name()` بافر مقداردهی/NUL نشده می‌خوند؛ `strtok` روی env؛ نشتی `asprintf`؛ ساخت پوشه‌های `/sdcard/wrapper/*` حتی وقتی لاگ خاموشه.
- `GetPhysicalDeviceProperties2` در هر فراخوانی چند رشته leak می‌کرد (`asprintf`).
- null-descriptor resourceها در `DestroyDevice` آزاد نمی‌شدن؛ hash-tableها leak می‌شدن.

## ۴) متغیرهای محیطی
- **حذف**: `ADRENOTOOLS_*`, `WRAPPER_EMULATE_BCN`, `WRAPPER_ASTC_BLOCK`, `WRAPPER_BCN_*`, `WRAPPER_USE_BCN_CACHE`, `WRAPPER_MARK_BCN`, `WRAPPER_NO_BCN_THREAD`, `WRAPPER_CACHE_PATH`, `WRAPPER_RESOURCE_TYPE`, `WRAPPER_DMAHEAP_CACHED`, `WRAPPER_DEVICE_NAME/ID`, `WRAPPER_VENDOR_ID`, `WRAPPER_DRIVER_ID`، و گزینه‌ی لاگ `bcn`.
- **جدید**: `WRAPPER_PIPELINE_CACHE`, `WRAPPER_PIPELINE_CACHE_DIR`, `WRAPPER_SUBGROUP_OPS`, `WRAPPER_DRM_NODE` (مسیر دستی نود DRM).
- بقیه (`WRAPPER_VK_VERSION`, `WRAPPER_DISABLE_PLACED`, `WRAPPER_DIAG`, …) همون‌طور موندن.

## ۵) نکات مهم
- **BCn**: روی Mali هایی که BC1–BC7 رو سخت‌افزاری ندارن، تکسچرهای DXT/BC (اکثر بازی‌های DXVK/vkd3d) دیگه کار نمی‌کنن یا بازی بالا نمیاد. این نتیجه‌ی مستقیم حذف BCn هست.
- ABI کرنل kbase (`VERSION_CHECK`, `SET_FLAGS`, `GET_GPUPROPS` و IDهای property) از حافظه/ساختار شناخته‌شده‌ی `mali_kbase_ioctl.h` نوشته شده؛ همه‌ی مراحل defensive هستن و خطا فقط یعنی «از اطلاعات Vulkan استفاده کن».
- `WRAPPER_SUBGROUP_OPS`: به‌صورت پیش‌فرض مثل قبل subgroup operations مخفی می‌مونه (رفتار تاریخی wrapper)؛ با `=1` واقعی‌ها نشون داده می‌شن.
- `vk_entrypoints.py` و `vk_extensions.py` توی zip ورودی symlink نبودن (متن ساده‌ی `../util/…`). همون رو دست نزدم؛ توی ریپوی خودت symlink واقعی نگه دار.
