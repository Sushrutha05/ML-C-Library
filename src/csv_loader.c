#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "csv_loader.h"

#define MAX_BUF 512

bool check_file_extension(const char *filename)
{
    if (!filename) return false;

    const char *dot = strrchr(filename, '.');
    return (dot && dot != filename && strcmp(dot, ".csv") == 0);
}

bool is_number(const char *str)
{
    if (!str) return false;

    if (*str == '-' || *str == '+') str++;

    bool has_digit = false;

    while (*str)
    {
        if (isdigit(*str)) has_digit = true;
        else if (*str == '.' || *str == '\n') {}
        else return false;

        str++;
    }

    return has_digit;
}

bool has_header(FILE *f)
{
    char buf[MAX_BUF];
    rewind(f);

    if (!fgets(buf, sizeof(buf), f)) return false;

    char temp[MAX_BUF];
    strncpy(temp, buf, sizeof(temp));
    temp[sizeof(temp)-1] = '\0';

    char *tok = strtok(temp, ",");
    return (tok == NULL || !is_number(tok));
}

int count_columns(FILE *f, bool header)
{
    char buf[MAX_BUF];
    rewind(f);

    if (!fgets(buf, sizeof(buf), f)) return 0;

    if (header && !fgets(buf, sizeof(buf), f)) return 0;

    int cols = 0;
    char *tok = strtok(buf, ",");

    while (tok)
    {
        cols++;
        tok = strtok(NULL, ",");
    }

    return cols;
}

int count_rows(FILE *f, bool header)
{
    char buf[MAX_BUF];
    int rows = 0;

    rewind(f);

    if (header) fgets(buf, sizeof(buf), f);

    while (fgets(buf, sizeof(buf), f))
        rows++;

    return rows;
}

static void parse_csv(FILE *f, bool header,
                      double *X, double *y,
                      int rows, int features)
{
    char buf[MAX_BUF];
    rewind(f);

    if (header) fgets(buf, sizeof(buf), f);

    int i = 0;

    while (fgets(buf, sizeof(buf), f))
    {
        char *tok = strtok(buf, ",");
        int j = 0;

        while (tok)
        {
            double val = atof(tok);

            if (j < features)
                X[i * features + j] = val;
            else
                y[i] = val;

            j++;
            tok = strtok(NULL, ",");
        }

        i++;
    }
}

Dataset load_csv(const char *filename)
{
    Dataset data = {0};

    if (!check_file_extension(filename))
        return data;

    FILE *f = fopen(filename, "r");
    if (!f) return data;

    bool header = has_header(f);

    data.cols = count_columns(f, header);
    data.rows = count_rows(f, header);
    data.features = data.cols - 1;

    data.X = malloc(data.rows * data.features * sizeof(double));
    data.y = malloc(data.rows * sizeof(double));

    if (!data.X || !data.y)
    {
        fclose(f);
        data.rows = 0;
        return data;
    }

    parse_csv(f, header, data.X, data.y, data.rows, data.features);

    fclose(f);
    return data;
}

void free_dataset(Dataset *data)
{
    if (!data) return;

    free(data->X);
    free(data->y);

    data->X = NULL;
    data->y = NULL;
}

void print_dataset(const Dataset *data)
{
    for (int i = 0; i < data->rows; i++)
    {
        printf("Row %d -> X: [", i);

        for (int j = 0; j < data->features; j++)
        {
            printf("%.2f", data->X[i * data->features + j]);
            if (j != data->features - 1)
                printf(", ");
        }

        printf("], y: %.2f\n", data->y[i]);
    }
}

/*
========================
|     MAIN FUNCTION     |
========================

#include <stdio.h>
#include "csv_loader.h"

int main()
{
    Dataset data = load_csv("so.csv");

    if (data.rows == 0)
    {
        printf("Failed to load dataset\n");
        return 1;
    }

    printf("Rows: %d, Cols: %d\n", data.rows, data.cols);

    print_dataset(&data);

    free_dataset(&data);

    return 0;
}
*/