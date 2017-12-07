
set(gadgetron_python_gadgets_files 
                           python/cine_auto_contouring.py 
                           python/IsmrmredImageArray_cine_auto_contouring.py
			   python/segmentation_settings.py
			   python/segmentation_tools.py
			   )

source_group(gadgets            FILES      ${gadgetron_python_gadgets_files})

install(FILES
    ${gadgetron_python_gadgets_files}
    DESTINATION ${GADGETRON_INSTALL_PYTHON_MODULE_PATH} COMPONENT main)
