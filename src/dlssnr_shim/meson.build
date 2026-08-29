remix_nvngx_src = files([
  'remix_nvngx.cpp',
])

# Note: the module name is load bearing. The DLSS-NR snippet rejects any call whose calling
# module path does not contain the substring "nvngx.dll" --- see the comment at the top of
# remix_nvngx.cpp. Do not rename this target.
remix_nvngx_dll = shared_library('remix_nvngx', remix_nvngx_src,
  name_prefix         : '',
  override_options    : ['cpp_std='+dxvk_cpp_std],
  install             : true
  )
