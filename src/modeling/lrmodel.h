#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Linked-list node definition
typedef struct Node {
    double X_1;
    double X_2;
    double X_3;
    double Y;
    struct Node *next;
} Node;




// Create a new node
extern Node* create_node(double X_1, double X_2, double X_3, double Y) ;
// Insert a new node
extern void insert_data(Node **head, double X_1, double X_2, double X_3, double Y);


extern void linear_regression(Node *head, double beta[4]);
