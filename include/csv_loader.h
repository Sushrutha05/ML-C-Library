#ifndef CSV_PARSER_H
#define CSV_PARSER_H

#include <stdbool.h>
#include <stdio.h>

// Struct to hold dataset
typedef struct {
    double *X;
    double *y;
    int rows;
    int cols;
    int features;
} Dataset;

// Utility
bool check_file_extension(const char *filename);
bool is_number(const char *str);

// Core API
bool has_header(FILE *f);
int count_columns(FILE *f, bool header);
int count_rows(FILE *f, bool header);

// Main function (important)
Dataset load_csv(const char *filename);

// Helper
void free_dataset(Dataset *data);
void print_dataset(const Dataset *data);

#endif