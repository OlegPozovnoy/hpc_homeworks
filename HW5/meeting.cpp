#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include <cstring>

int get_rand(int max);

int get_random_index(char *buf, int size, int proc_name);

const int NAME_LEN = 32;

int main(int argc, char **argv) {
    int prank, psize;

    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    char *buf = (char *) malloc(sizeof(char *)  * psize * NAME_LEN);

    int step = 1;
    int total_steps = 4;
    int send_to;
    if (prank == 0) {
        sprintf(buf, "{0 thread has name proc0}");
        send_to = get_random_index(buf, psize, prank);
        printf("\nSending message from thread %d to thread %d", prank, send_to);
        MPI_Ssend(buf, psize * NAME_LEN, MPI_CHAR, send_to, step, MPI_COMM_WORLD);
        MPI_Recv(buf, psize * NAME_LEN, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("\nProcess %d is finished", prank);
        return 0;
    }

    MPI_Recv(buf, psize * NAME_LEN, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    printf("\nReceived message from thread %d to thread %d", prank, prank);
    printf("\n%s", buf);
    char to_append[NAME_LEN]; sprintf(to_append, "{%d thread has name proc%d}", prank, psize);
    strcat(buf,to_append);
    send_to = get_random_index(buf, psize, prank);
    if (send_to >=0){
        printf("\nSending message from thread %d to thread %d", prank, send_to);
        printf("\n%s", buf);
        MPI_Ssend(buf, psize * NAME_LEN, MPI_CHAR, send_to, step, MPI_COMM_WORLD);
    } else{
        printf("\nNo more receivers. Sending to 0 process termination signal.");
        MPI_Ssend(buf, psize * NAME_LEN, MPI_CHAR, 0, step, MPI_COMM_WORLD);
    }
    printf("\nProcess %d is finished", prank);
    return 0;

}

int get_rand(int max) {
    srand(time(NULL));   // Initialization, should only be called once.
    return rand() % max;
}

int get_random_index(char *buf, int size, int proc_name) {
    int available = 0;
    int n = 0;
    int index;

    bool is_received[size];
    for(int i =0;i<size; i++)
        is_received[i] = false;

    is_received[proc_name] = true;
    printf("\n%s", buf);

    int total = 0;
    while (sscanf(buf + total, "%*[^0-9]%d%*[^}]%n", &index,&n) > 0) {
        total += n;
        is_received[index] = true;
    }

    for (int i = 0; i < size; i++)
        if (!is_received[i])
            available++;

    if (available == 0) {
        printf("\nno more available threads");
        return -1;
    }

    int send_to = get_rand(available) + 1;
    printf("\n%d process available for sending. Send to %d available slot.", available, send_to);

    for (int i = 0; i < size; i++) {
        if (!is_received[i]) {
            send_to--;
        }
        if (send_to == 0)
            return i;
    }
    return 0;
}
