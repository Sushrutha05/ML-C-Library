#include <string.h>
#include <stdbool.h>

bool has_file_extension(const char *filename, const char *ext)
{
    if (!filename) return false;

    const char *dot = strrchr(filename, '.');
    return (dot && dot != filename && *(dot+1) != '\0' && strcmp(dot+1, ext) == 0);
}