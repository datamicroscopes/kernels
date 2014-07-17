message(STATUS "Finding microscopes-mixturemodel")

execute_process(
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_in_venv_like.sh microscopes_mixturemodel microscopes
    OUTPUT_VARIABLE MICROSCOPES_MIXTUREMODEL_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(MICROSCOPES_MIXTUREMODEL_ROOT)
    set(MICROSCOPES_MIXTUREMODEL_INCLUDE_DIRS ${MICROSCOPES_MIXTUREMODEL_ROOT}/include)
    set(MICROSCOPES_MIXTUREMODEL_LIBRARY_DIRS ${MICROSCOPES_MIXTUREMODEL_ROOT}/lib)
    set(MICROSCOPES_MIXTUREMODEL_LIBRARIES microscopes_mixturemodel)
    set(MICROSCOPES_MIXTUREMODEL_FOUND true)
else()
    message(STATUS "could not locate microscopes_mixturemodel") 
    set(MICROSCOPES_MIXTUREMODEL_FOUND false)
endif()
