#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define FILENAMEMAX   100

int main() 
{
    struct rlimit lim;
    long int i;
    int fd;
    char filename[FILENAMEMAX];

    getrlimit(RLIMIT_NOFILE, &lim);
    printf("Before set soft limit, Soft limit = %ld, Hard limit = %ld\n", lim.rlim_cur, lim.rlim_max);

    lim.rlim_cur = 8888;
    setrlimit(RLIMIT_NOFILE, &lim);
    getrlimit(RLIMIT_NOFILE, &lim);
    printf("After set soft limit, Soft limit = %ld, Hard limit = %ld\n", lim.rlim_cur, lim.rlim_max);

    int count = 0;
    for (i = 0; i < lim.rlim_max; ++i) {
        snprintf(filename, sizeof(filename), "%d_%ld", getpid(), i);
        fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0664);

        if (fd < 0) {
            fprintf(stderr, "open file %s error\n", filename);
            exit(1);
        }
        else {
            if (count < 3) {
                printf("open file %s success\n", filename);
                count++;
            }
        }
        // close(fd);
    }
}