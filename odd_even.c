#include <stdio.h>
#include <time.h>

#define N 8

// Serial odd-even transposition sort
void oddEvenSort(int arr[], int n){
    for(int phase = 0; phase < n; phase++){
        // Even phase when phase%2==0, Odd phase when phase%2==1
        for(int i = (phase % 2); i < n-1; i += 2){
            if(arr[i] > arr[i+1]){
                int tmp = arr[i];
                arr[i] = arr[i+1];
                arr[i+1] = tmp;
            }
        }
    }
}

int main() {
    int arr[N] = {5,2,8,3,1,6,4,7};

    clock_t start = clock();
    oddEvenSort(arr, N);
    clock_t end = clock();

    printf("Sorted array: ");
    for(int i=0;i<N;i++) printf("%d ", arr[i]);
    printf("\n");

    double time_ms = (double)(end-start) * 1000 / CLOCKS_PER_SEC;
    printf("Serial execution time: %f ms\n", time_ms);

    return 0;
}
