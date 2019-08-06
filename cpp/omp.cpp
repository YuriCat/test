#include <stdio.h>
#include <omp.h>

int main() {
  #pragma omp parallel for
  for (int i = 0; i < 5; i++) {
    #pragma omp critical
    {
      printf("Thread No: %d\n", i);
    }
  }
}
