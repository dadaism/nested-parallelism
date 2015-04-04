#ifndef __UTIL_H_
#define __UTIL_H_

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

// return current timestamp in seconds.
inline double gettime_ms() {
        struct timeval t;
        gettimeofday(&t,NULL);
        return (t.tv_sec+t.tv_usec*1e-6)*1000;
}

#endif //UTIL_H_


