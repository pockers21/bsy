################################################################################################
# Removes duplicates from list(s)
# Usage:
#   list_unique(<list_variable> [<list_variable>] [...])
macro(list_unique)
  foreach(__lst ${ARGN})
    if(${__lst})
      list(REMOVE_DUPLICATES ${__lst})
    endif()
  endforeach()
endmacro()

################################################################################################
# Converts all paths in list to absolute
# Usage:
#   convert_absolute_paths(<list_variable>)
function(convert_absolute_paths variable)
  set(__dlist "")
  foreach(__s ${${variable}})
    get_filename_component(__abspath ${__s} ABSOLUTE)
    list(APPEND __list ${__abspath})
  endforeach()
  set(${variable} ${__list} PARENT_SCOPE)
endfunction()