find_package(Doxygen)

if (Doxygen_FOUND)
    file(GLOB_RECURSE HEADERS ../include/puffinn/*.hpp)
    set(DOXYGEN_INDEX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/html/index.html)

    set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/include/puffinn)
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
    set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
    set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

    file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
    add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                       DEPENDS ${HEADERS}
                       COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       MAIN_DEPENDENCY ${DOXYFILE_OUT}
                       COMMENT "Running Doxygen")
    add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

    find_package(Sphinx REQUIRED)

    set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
    set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
    set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

    add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
        COMMAND ${SPHINX_EXECUTABLE} -b html -Dbreathe_projects.PUFFINN=${DOXYGEN_OUTPUT_DIR}/xml ${SPHINX_SOURCE} ${SPHINX_BUILD}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
        ${DOXYGEN_INDEX_FILE}
        MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
        COMMENT "Running Sphinx")

    add_custom_target(Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})

    include(GNUInstallDirs)
    install(DIRECTORY ${SPHINX_BUILD} DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif()
