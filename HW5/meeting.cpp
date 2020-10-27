#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include <cstring>

int get_rand(int max);
bool* get_finished_process(char* buf, const int &size);
int get_random_index(char *buf, const int &size, const int &proc_name);

const int NAME_LEN = 32;

int main(int argc, char **argv) {
    int prank, psize;

    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    char *buf = (char *) malloc(sizeof(char *)  * psize * NAME_LEN);

    int current_tag = 0;
    int total_steps = 1;
    int send_to;
    if (prank == 0) {
        sprintf(buf, "{0 thread has name proc0}");
        send_to = get_random_index(buf, psize, prank);
        printf("\nSending message from thread %d to thread %d", prank, send_to);
        MPI_Ssend(buf, psize * NAME_LEN, MPI_CHAR, send_to, current_tag, MPI_COMM_WORLD);
    }

    MPI_Recv(buf, psize * NAME_LEN, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    if (strcmp(buf, "finish")==0){
        printf("\nProcess %d received termination command and will be finished", prank);
        return 0;
    }

    printf("\nReceived message from thread %d to thread %d", status.MPI_SOURCE, prank);
    printf("\n%s", buf);
    char to_append[NAME_LEN]; sprintf(to_append, "{%d thread has name proc%d}", prank, prank);
    strcat(buf,to_append);

    current_tag = status.MPI_TAG + 1;
    if (current_tag >= total_steps){
        printf("\nTotal number of messages exceeded N, sending termination command");
        bool* finished = get_finished_process(buf, psize);
        finished[0] = false;
        sprintf(buf, "finish");
        for (int i=0;i<psize;i++){
            if (!finished[i]) {
                printf("\nSending termination command to proc %d", i);
                MPI_Ssend(buf, psize * NAME_LEN, MPI_CHAR, i, current_tag, MPI_COMM_WORLD);
            }
        }
        free(finished);
    } else {
        send_to = get_random_index(buf, psize, prank);
        if (send_to >= 0) {
            printf("\nSending message from thread %d to thread %d", prank, send_to);
            printf("\n%s", buf);
            MPI_Ssend(buf, psize * NAME_LEN, MPI_CHAR, send_to, current_tag, MPI_COMM_WORLD);
        } else {
            printf("\nNo more receivers. Sending to 0 process termination signal.");
            sprintf(buf, "finish");
            MPI_Ssend(buf, psize * NAME_LEN, MPI_CHAR, 0, current_tag, MPI_COMM_WORLD);
        }
    }
    printf("\nProcess %d is finished", prank);
    return 0;

}

int get_rand(int max) {
    srand(time(NULL));   // Initialization, should only be called once.
    return rand() % max;
}

bool* get_finished_process(char* buf, const int &size){
    bool* is_finshed = (bool*)calloc(size, sizeof(bool));
    int total = 0;
    int n = 0;
    int index;

    while (sscanf(buf + total, "%*[^0-9]%d%*[^}]%n", &index,&n) > 0) {
        total += n;
        is_finshed[index] = true;
    }
    return is_finshed;
}

int get_random_index(char* buf, const int &size, const int &proc_name) {
    int available = 0;
    bool* is_received = get_finished_process(buf, size);

    for (int i = 0; i < size; i++)
        if (!is_received[i])
            available++;

    if (available == 0) {
        printf("\nNo more available threads");
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

    free(is_received);

    return 0;
}
