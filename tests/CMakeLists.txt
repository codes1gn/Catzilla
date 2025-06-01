# Enable testing
enable_testing()

# Initialize TEST_BINARIES
set(TEST_BINARIES "")

# Add test utils first
add_subdirectory(test_utils)

# Add unittests
add_subdirectory(unittests)

# Add recipe tests
add_subdirectory(recipetests)

# Create a custom target that depends on all tests
add_custom_target(catz_tests
    DEPENDS 
    recipe_tests 
    ${TEST_BINARIES}
)
