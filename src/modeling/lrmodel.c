#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lrmodel.h"
#define lambda 1e-5;

// Create a new node
Node* create_node(double X_1, double X_2, double X_3, double Y) {
    Node *new_node = (Node *)malloc(sizeof(Node));
    new_node->X_1 = X_1;
    new_node->X_2 = X_2;
    new_node->X_3 = X_3;
    new_node->Y = Y;
    new_node->next = NULL;
    return new_node;
}

// Get the linked-list length
int get_list_length(Node *head) {
    int length = 0;
    Node *current = head;
    while (current != NULL) {
        length++;
        current = current->next;
    }
    return length;
}

// Append data to the linked list
void insert_data(Node **head, double X_1, double X_2, double X_3, double Y) {
    Node* newNode = create_node(X_1, X_2, X_3, Y);
    if (*head == NULL) {
        *head = newNode;
    } else {
        Node* temp = *head;
        while (temp->next != NULL) {
            temp = temp->next;
        }
        temp->next = newNode;
    }
}

// Transpose a matrix
void transpose(double mat[], double result[], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j * rows + i] = mat[i * cols + j];
        }
    }
}

// Matrix multiplication
void mat_mult(double mat1[], double mat2[], double result[], int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            result[i * p + j] = 0;
            for (int k = 0; k < n; k++) {
                result[i * p + j] += mat1[i * n + k] * mat2[k * p + j];
            }
        }
    }
}

// Invert Xt * X using simplified Gaussian elimination
int inverse_matrix(double matrix[], double result[], int n) {
    double temp[n][n*2];  // Augmented matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            temp[i][j] = matrix[i * n + j];
            temp[i][j + n] = (i == j) ? 1 : 0;  // Identity matrix on the right
        }
    }

    // Gaussian elimination
    for (int i = 0; i < n; i++) {
        if (temp[i][i] == 0) {
            return 0;  // A zero pivot prevents inversion with this implementation
        }

        double pivot = temp[i][i];
        for (int j = 0; j < 2 * n; j++) {
            temp[i][j] /= pivot;
        }

        for (int j = 0; j < n; j++) {
            if (i != j) {
                double factor = temp[j][i];
                for (int k = 0; k < 2 * n; k++) {
                    temp[j][k] -= temp[i][k] * factor;
                }
            }
        }
    }

    // Extract the inverse matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i * n + j] = temp[i][j + n];
        }
    }

    return 1;  // Inversion succeeded
}

// Compute the regression coefficients
void linear_regression(Node *head, double beta[4]) {
    // Compute the linked-list length N
    int N = get_list_length(head);

    // Allocate the design matrix X and response vector Y
    double *X = (double *)malloc(N * 4 * sizeof(double));  // Includes the intercept
    double *Y = (double *)malloc(N * sizeof(double));
    double *Xt = (double *)malloc(4 * N * sizeof(double));  // Transpose of X
    double *XtX = (double *)malloc(4 * 4 * sizeof(double));  // Xt * X
    double *XtX_inv = (double *)malloc(4 * 4 * sizeof(double));  // Inverse of Xt * X
    double *XtY = (double *)malloc(4 * sizeof(double));  // Xt * Y
    // double *beta = (double *)malloc(4 * sizeof(double));  // Regression coefficients

    Node *current = head;
    // Populate X and Y
    for (int i = 0; i < N; i++) {
        X[i * 4 + 0] = 1;  // Intercept term
        X[i * 4 + 1] = current->X_1;
        X[i * 4 + 2] = current->X_2;
        X[i * 4 + 3] = current->X_3;
        Y[i] = current->Y;
        current = current->next;
    }

    // Compute Xt (the transpose of X)
    transpose(X, Xt, N, 4);

    // Compute Xt * X
    mat_mult(Xt, X, XtX, 4, N, 4);

    for (int i = 0; i < 4; i++) {
        XtX[i * 4 + i] += lambda; // Add lambda to the diagonal
    }

    // Compute Xt * Y
    for (int i = 0; i < 4; i++) {
        XtY[i] = 0;
        for (int j = 0; j < N; j++) {
            XtY[i] += Xt[i * N + j] * Y[j];
        }
    }

    // Invert Xt * X
    if (!inverse_matrix(XtX, XtX_inv, 4)) {
        printf("Matrix inversion failed; cannot compute regression coefficients.\n");
        free(X);
        free(Y);
        free(Xt);
        free(XtX);
        free(XtX_inv);
        free(XtY);
        // free(beta);
        return;
    }

    // Compute beta from the inverse of Xt * X and Xt * Y
    for (int i = 0; i < 4; i++) {
        beta[i] = 0;
        for (int j = 0; j < 4; j++) {
            beta[i] += XtX_inv[i * 4 + j] * XtY[j];
        }
    }


    // Free allocated memory
    free(X);
    free(Y);
    free(Xt);
    free(XtX);
    free(XtX_inv);
    free(XtY);
    // free(beta);
}

// int main() {
//     Node *head = NULL;

//     // Example data
//     double X_1[] = {588, 1176, 2352, 4704, 9408, 18816, 37632, 1200, 2400, 4800, 9600, 19200, 38400, 76800, 3072, 6144, 12288, 24576, 49152, 98304, 196608, 4800, 9600, 19200, 38400, 76800, 153600, 307200};
//     double X_2[] = {11.76, 23.52, 47.04, 94.08, 188.16, 376.32, 752.64, 11.88118812, 23.76237624, 47.52475248, 95.04950495, 190.0990099, 380.1980198, 760.3960396, 11.95330739, 23.90661479, 47.81322957, 95.62645914, 191.25291829, 382.50583658, 765.01167315, 11.97007481, 23.94014963, 47.88029925, 95.7605985, 191.52119701, 383.04239401, 766.08478803};
//     double X_3[] = {64, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256};
//     double Y[] = {3.6688304, 6.33579969, 8.81930828, 15.89013815, 28.59649897, 55.26623964, 106.5889359, 5.11982679, 8.30752373, 15.17188549, 28.83654594, 55.6793642, 107.7313852, 209.9392509, 10.42109489, 18.68229628, 35.30418634, 67.3269248, 131.7667437, 259.3008351, 521.1464882, 14.61896896, 28.13376188, 54.63783264, 105.8555222, 206.7464566, 409.3927217, 823.2596278};
//     double beta[4];
//     for (int i = 0; i < 28; i++)
//     {
//         insert_data(&head, X_1[i], X_2[i], X_3[i], Y[i]);

//         linear_regression(head, beta);
//         // Print regression coefficients
//         // printf("beta_0: %f\n", beta[0]);
//         // printf("beta_1: %f\n", beta[1]);
//         // printf("beta_2: %f\n", beta[2]);
//         // printf("beta_3: %f\n", beta[3]);
//         // Print the regression equation
//         printf("Model: Y = %.4f + %.4f * X1 + %.4f * X2 + %.4f * X3\n", beta[0], beta[1], beta[2], beta[3]);

//         // double err_sum = 0;
//         // for(int i = 0; i<28; i++){
//         //     double y_pred = X_1[i] * beta[1] + X_2[i] * beta[2] + X_3[i] * beta[3] + beta[0];
//         //     double err = fabs(y_pred - Y[i]) / Y[i];
//         //     err_sum += err;
//         // }
//         // printf("------------------%d-----------------------\n", i);
// 		// printf("Err_mean = %4f\n", (err_sum / (i+1)));
//          // Traverse the linked list to compute the sums
//         double err_sum1 = 0;
// 		Node* temp = head;
// 		while (temp != NULL) {
// 			double y_pred = temp->X_1 * beta[1] + temp->X_2 * beta[2] + temp->X_3 * beta[3] + beta[0];
// 			double err = fabs(y_pred - temp->Y) /  temp->Y;
// 			err_sum1 += err;
//             temp = temp->next;
// 		}
//         printf("------------------%d-----------------------\n", i);
// 		printf("Err_mean = %4f\n", (err_sum1 / (i+1)));

//     }



//     return 0;
// }
