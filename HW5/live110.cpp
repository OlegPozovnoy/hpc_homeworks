#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include <cstring>
#include <omp.h>

int get_rand(int max, int seed);

bool *get_finished_process(char *buf, const int &size);

int get_random_index(char *buf, const int &size, const int &proc_name);

inline int rule110(const int &prev, const int &current, const int &next);

inline int rule184(const int &prev, const int &current, const int &next);

const int GAME_SIZE = 1000;
const int STEPS_NUM = 1000;
using namespace std;

//2) В зависимости от значений: левого соседа, себя, правого соседа на следующем шаге клетка либо меняет значение, либо остается той же. Посмотрите, например, что значит Rule110 (https://en.wikipedia.org/wiki/Rule_110)
//Сделайте периодические и непериодические граничные условия (3 балла)

int main(int argc, char **argv) {
    int prank, psize;
    omp_set_num_threads(2);
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    bool periodic = false;

    int (*rule)(const int &, const int &, const int &);
    rule = rule110;

    for (int i = 0; i < argc; ++i)
        if (strcmp(argv[i], "rule184") == 0) {
            if (prank == 0)
                printf("\nSwitching from rule 110 to rule184");
            rule = rule184;
        } else if (strcmp(argv[i], "periodic") == 0) {
            if (prank == 0)
                printf("\nSwitching to periodic mode");
            periodic = true;
        }
    if (prank == 0)
        printf("\nWidth=%d, Height=%d", GAME_SIZE, STEPS_NUM);
    double time_elapsed = MPI_Wtime();

    int piece_size = (prank + 1) * GAME_SIZE / psize - prank * GAME_SIZE / psize;
    //слева и справа ячейки для ghost cells
    int *buf = (int *) malloc(sizeof(int) * (piece_size + 2) * STEPS_NUM);

    // инициализируем нашу часть массива
    for (int i = 1; i < piece_size + 1; i++) {
        buf[i] = get_rand(2, i + prank * psize * 107);
    }
    // инициализируем граничные клетки, если режим periodic-значения игнорируем
    int left_border = get_rand(2, prank * psize * 999);
    int right_border = get_rand(2, prank * psize * 1001);
    if (!periodic) {
        buf[0] = left_border;
        buf[piece_size + 1] = right_border;
    }
    // разошлем соседям наши края и получим от них их края
    //printf("\n process %d, out of %d", prank, psize);
    for (int row = 0; row < STEPS_NUM - 1; row++) {
        int send_to_left = (prank == 0) ? psize - 1 : prank - 1;
        int send_to_right = (prank == psize - 1) ? 0 : prank + 1;

        if (periodic && psize > 1) {
            if (prank != 0)
                MPI_Send(&buf[(piece_size + 2) * row + 1], 1, MPI_INT, send_to_left, row, MPI_COMM_WORLD);

            if (prank != psize - 1)
                MPI_Send(&buf[(piece_size + 2) * row + piece_size], 1, MPI_INT, send_to_right, row, MPI_COMM_WORLD);

            if (prank != 0)
                MPI_Recv(&buf[(piece_size + 2) * row], 1, MPI_INT, send_to_left, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (prank != psize - 1)
                MPI_Recv(&buf[(piece_size + 2) * row + piece_size + 1], 1, MPI_INT, send_to_right, MPI_ANY_TAG,
                         MPI_COMM_WORLD,
                         &status);
        } else if (periodic && psize == 1) {
            buf[(piece_size + 2) * (row + 1)] = buf[(piece_size + 2) * (row + 2) - 2];
            buf[(piece_size + 2) * (row + 2) - 1] = buf[(piece_size + 2) * (row + 1) + 1];
        } else {// (!periodic)
            buf[(piece_size + 2) * (row + 1)] = left_border;
            buf[(piece_size + 2) * (row + 2) - 1] = right_border;
        }

        int omp_i;
#pragma omp parallel default(none) private(omp_i) shared(buf, row, rule, piece_size)
        {
            //int tid = omp_get_thread_num();
            //printf("\ntid=%d",tid);
#pragma omp parallel for
            for (omp_i = (piece_size + 2) * (row + 1) + 1; omp_i < (piece_size + 2) * (row + 2) - 1; omp_i++) {

                buf[omp_i] = rule(buf[omp_i - 1 - (piece_size + 2)], buf[omp_i - (piece_size + 2)],
                                  buf[omp_i + 1 - (piece_size + 2)]);
            }
        }
    }

    FILE *pFile;
    char filename[16];
    sprintf(filename, "out%d.txt", prank);
    pFile = fopen(filename, "w");
//printf("\n process %d", prank);
    for (int row = 0; row < STEPS_NUM; row++) {
        fprintf(pFile, "\n");
        int from = periodic ? 1 : 0;
        int to = periodic ? piece_size + 1 : piece_size + 2;
        for (int i = from; i < to; i++) {
            fprintf(pFile, "%d", buf[row * (piece_size + 2) + i]);
        }
    }
    fclose(pFile);
    free(buf);
    MPI_Barrier(MPI_COMM_WORLD);

    if (prank == 0) {
        // измерим здесь тк ввод вывод все таки не алгоритм
        time_elapsed = MPI_Wtime() - time_elapsed;
        char command[256];
        strcpy(command, "paste ");
        for (int i = 0; i < psize; i++) {
            char filename[16];
            sprintf(filename, "out%d.txt ", i);
            strcat(command, filename);
        }
        strcat(command, "-d '' >> rule.out");//>> rule.out
        //printf("\n%s", command);
        system(command);
        printf("\ntime_elapsed %f", time_elapsed);
    }

    MPI_Finalize();
    return 0;
}

int get_rand(int max, int seed) {
    srand(time(NULL) + seed);
    return rand() % max;
}

//111	110	101	100	011	010	001	000
//0	    1	1	0	1	1	1	0
inline int rule110(const int &prev, const int &current, const int &next) {
    if ((prev == current && current == next) || (prev == 1 && current == 0 && next == 0))
        return 0;
    else
        return 1;
}

//Rule184
//current pattern	        111	110	101	100	011	010	001	000
//new state for center cell	1	0	1	1	1	0	0	0
inline int rule184(const int &prev, const int &current, const int &next) {
    int sum = (prev << 2) + (current << 1) + next;
    if (sum == 6 || sum <= 2)
        return 0;
    else
        return 1;
}

