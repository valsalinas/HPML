#include <stdio.h>

void oddEvenSort(int *arr, int n) {
    for(int i=0; i<n; i++){
        // Even phase
        for(int j=0; j<n-1; j+=2){
            if(arr[j] > arr[j+1]){
                int tmp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = tmp;
            }
        }
        // Odd phase
        for(int j=1; j<n-1; j+=2){
            if(arr[j] > arr[j+1]){
                int tmp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = tmp;
            }
        }
    }
}

int main() {
    int arr[8] = {5, 3, 8, 1, 4, 7, 2, 6};
    int n = 8;

    printf("Before sorting:\n");
    for(int i=0;i<n;i++) printf("%d ", arr[i]);
    printf("\n");

    oddEvenSort(arr, n);

    printf("After sorting:\n");
    for(int i=0;i<n;i++) printf("%d ", arr[i]);
    printf("\n");
    return 0;
}
