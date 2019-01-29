#include <string.h>
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <malloc.h>
//Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencilSerial(const int nx, const int ny, const float *restrict  image, float *restrict tmp_image);
void stencilMaster(const int localCols, const int ny, const float *restrict image, float *restrict tmp_image);
void stencilMidWorker(const int localCols, const int ny, const float *restrict image, float *restrict tmp_image);
void stencilFinWorker(const int localCols, const int ny, const float *restrict image, float *restrict tmp_image);

void init_image(const int nx, const int ny, float *  image, float *  tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
double wtime(void);

int main(int argc, char *argv[]) {
  //Parameters useful for MPI related procedures
  int left;
  int right;
  int rank;
  int size;
  int flag;
  double tic;
  double toc;
  int localCols = 0;
  int remainderCols = 0;
  int haloCount = 0;
  enum bool {FALSE,TRUE}; /* enumerated type: false = 0, true = 1 */
  //Arrays for storing images/sub images
  float *image;
  float *tmp_image;
  float *sub_image;
  float *sub_tmp_image;


  MPI_Status status;     /* struct used by MPI_Recv */
  MPI_Request request;
  //Initialise MPI Environment
  MPI_Init(&argc, &argv);
  /* check whether the initialisation was successful */
  MPI_Initialized(&flag);
  //Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }
  //Check initialization success
  if(flag != TRUE){
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }
  //Initialise rank (of process) and size (no. of processes)
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  //Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);
  //If size is 1, no point parallelising
  if(size == 1){
    //Allocate the image
    image =  malloc(sizeof(float)*nx*ny );
    tmp_image =  malloc(sizeof(float)*nx*ny );

    //Set the input image
    init_image(nx, ny, image, tmp_image);

    //Call the stencil kernel
    tic = wtime();
    for (int t = 0; t < niters; ++t) {
      stencilSerial(nx, ny, image, tmp_image);
      stencilSerial(nx, ny, tmp_image, image);
    }
    toc = wtime();
    //Output
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc-tic);
    printf("------------------------------------\n");

    output_image(OUTPUT_FILE, nx, ny, image);
     free(image);
     free(tmp_image);
  }
  //i.e. if there are multiple workers
  else{
    //Initialise arrays for scatterv
    int *sendcounts =  malloc(size*sizeof(int)  );
    int *displs =  malloc(size*sizeof(int)  );
    //Initiliase left and right neighbours for MPI communication
    right = (rank + 1) % size;
    left = (rank == 0) ? (rank + size - 1) : (rank - 1);
    //Initialise local columns, accounting for an uneven distribution
    if(rank != MASTER){
      localCols = nx / size;
    }
    if((nx % size != 0) && (rank == size -1)){
      remainderCols = nx % size;
    }

    //Initialise memory
    if(rank == MASTER){
      image =  malloc(sizeof(float)*nx*ny  );
      tmp_image =  malloc(sizeof(float)*nx*ny  );
      init_image(nx, ny, image, tmp_image);
    }
    else if(rank != size - 1){
      haloCount = 2*ny;
      sub_image =   malloc(sizeof(float)*(ny*(localCols+remainderCols)+haloCount)  );
      sub_tmp_image =   malloc(sizeof(float)*(ny*(localCols+remainderCols)+haloCount)  );
    }
    else{
      haloCount = ny;
      sub_image =   malloc(sizeof(float)*(ny*(localCols+remainderCols)+haloCount)  );
      sub_tmp_image =   malloc(sizeof(float)*(ny*(localCols+remainderCols)+haloCount) );
    }
    //Initialise image and distribute
    sendcounts[MASTER] = 0;
    displs[MASTER] = 0;
    int offset = nx/size * ny - ny;
    for(int i = 1; i < size - 1; i++){
      sendcounts[i] = nx/size * ny + (2*ny);
      displs[i] = offset;
      offset = offset + nx/size * ny;
    }
    sendcounts[size-1] = (nx/size + (nx % size)) * ny + ny;
    displs[size-1] = offset;

    MPI_Scatterv(image,sendcounts,displs,MPI_FLOAT,sub_image,(localCols+remainderCols)*ny + haloCount,MPI_FLOAT,MASTER,MPI_COMM_WORLD);
    MPI_Scatterv(tmp_image,sendcounts,displs,MPI_FLOAT,sub_tmp_image,(localCols+remainderCols)*ny + haloCount,MPI_FLOAT,MASTER,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    tic = wtime();
    if(rank == MASTER){
      //Perform stencil
      haloCount = ny;
      localCols = nx/size;
      for(int i = 0; i < niters; i++){
        stencilMaster(localCols, ny, image, tmp_image);
        MPI_Sendrecv(tmp_image+(ny*localCols)-ny,ny,MPI_FLOAT,right,0,tmp_image+(ny*localCols),ny,MPI_FLOAT,right,0,MPI_COMM_WORLD,&status);
        stencilMaster(localCols, ny, tmp_image, image);
        MPI_Sendrecv(image+(ny*localCols)-ny,ny,MPI_FLOAT,right,0,image+(ny*localCols),ny,MPI_FLOAT,right,0,MPI_COMM_WORLD,&status);
      }
    }
    //Receive grid if worker
    else if(rank != size-1){
      if(rank % 2){
        for(int i = 0; i < niters; i++){
          stencilMidWorker(localCols, ny, sub_image+ny, sub_tmp_image+ny);
          MPI_Sendrecv(sub_tmp_image+ny,ny,MPI_FLOAT,left,0,sub_tmp_image,ny,MPI_FLOAT,left,0,MPI_COMM_WORLD,&status);
          MPI_Sendrecv(sub_tmp_image+(localCols*ny),ny,MPI_FLOAT,right,0,sub_tmp_image+(localCols*ny)+ny,ny,MPI_FLOAT,right,0,MPI_COMM_WORLD,&status);
          stencilMidWorker(localCols, ny, sub_tmp_image+ny, sub_image+ny);
          MPI_Sendrecv(sub_image+ny,ny,MPI_FLOAT,left,0,sub_image,ny,MPI_FLOAT,left,0,MPI_COMM_WORLD,&status);
          MPI_Sendrecv(sub_image+(localCols*ny),ny,MPI_FLOAT,right,0,sub_image+(localCols*ny)+ny,ny,MPI_FLOAT,right,0,MPI_COMM_WORLD,&status);
        }
      }
      else{
        for(int i = 0; i < niters; i++){
          stencilMidWorker(localCols, ny, sub_image+ny, sub_tmp_image+ny);
          MPI_Sendrecv(sub_tmp_image+(localCols*ny),ny,MPI_FLOAT,right,0,sub_tmp_image+(localCols*ny)+ny,ny,MPI_FLOAT,right,0,MPI_COMM_WORLD,&status);
          MPI_Sendrecv(sub_tmp_image+ny,ny,MPI_FLOAT,left,0,sub_tmp_image,ny,MPI_FLOAT,left,0,MPI_COMM_WORLD,&status);
          stencilMidWorker(localCols, ny, sub_tmp_image+ny, sub_image+ny);
          MPI_Sendrecv(sub_image+(localCols*ny),ny,MPI_FLOAT,right,0,sub_image+(localCols*ny)+ny,ny,MPI_FLOAT,right,0,MPI_COMM_WORLD,&status);
          MPI_Sendrecv(sub_image+ny,ny,MPI_FLOAT,left,0,sub_image,ny,MPI_FLOAT,left,0,MPI_COMM_WORLD,&status);
        }
      }

    }
    else{
        for(int i = 0; i < niters; i++){
          stencilFinWorker(localCols+remainderCols, ny, sub_image+ny, sub_tmp_image+ny);
          MPI_Sendrecv(sub_tmp_image+ny,ny,MPI_FLOAT,left,0,sub_tmp_image,ny,MPI_FLOAT,left,0,MPI_COMM_WORLD,&status);
          stencilFinWorker(localCols+remainderCols, ny, sub_tmp_image+ny, sub_image+ny);
          MPI_Sendrecv(sub_image+ny,ny,MPI_FLOAT,left,0,sub_image,ny,MPI_FLOAT,left,0,MPI_COMM_WORLD,&status);
        }

    }
    MPI_Barrier(MPI_COMM_WORLD);
    toc = wtime();
    int subOffset = (rank == 0) ? 0 : ny;
    displs[0] = 0;
    sendcounts[0] = 0;
    for(int i = 1; i < size-1; i++){
      displs[i] = i * localCols * ny;
      sendcounts[i] = ny*(localCols);
    }
    displs[size-1] = (size-1) * localCols * ny;
    sendcounts[size-1] = ny*(localCols + (nx % size));
    if(rank == MASTER){
      localCols = 0;
    }


    MPI_Gatherv(sub_image+subOffset, ny*(localCols+remainderCols), MPI_FLOAT, image, sendcounts, displs, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    if(rank == MASTER){

      //Output
      printf("------------------------------------\n");
      printf(" dimensions: %d x %d runtime: %lf s\n", nx,ny,toc-tic);
      printf("------------------------------------\n");

      output_image(OUTPUT_FILE, nx, ny, image);
    }
    if(rank==MASTER){
       free(image);
       free(tmp_image);
    }
    else{
       free(sub_image);
       free(sub_tmp_image);
    }

     free(sendcounts);
     free(displs);

  }
  MPI_Finalize();
}

void stencilSerial(const int nx, const int ny, const float *restrict image, float *restrict tmp_image){
  //__assume_aligned(image  );
  //__assume_aligned(tmp_image  );
  //__assume(nx%16==0);
  //__assume(ny%16==0);
    //Top left corner
	tmp_image[0] = image[0] * 0.6f + image[1] * 0.1f + image[ny] * 0.1f;
	//Left Column
	for(int i = 1; i < ny-1; i++){
		tmp_image[i] = image[i] * 0.6f + image[i-1] * 0.1f + image[i+1] * 0.1f + image[i+ny] * 0.1f;
	}

	//Bottom let corner
	tmp_image[ny-1] = image[ny-1] * 0.6f + image[ny-2] * 0.1f + image[2*ny-1] * 0.1f;

    //All middle pixels/edges
    for(int i = 1; i < nx - 1; i++){
      //printf("%d\n",i);
        //Left column bit
        tmp_image[ny*i] = image[ny*i] * 0.6f + image[ny*i-ny] * 0.1f + image[ny*i+1] * 0.1f + image[ny*i+ny] * 0.1f;
        //Middle pixels
        for(int j = 1; j < ny - 1; j++){
            tmp_image[j+i*ny] = image[j+i*ny] * 0.6f + image[j+i*ny-1] * 0.1f + image[j+i*ny+1] * 0.1f + image[j+i*ny-ny] * 0.1f + image[j + i*ny+ny] * 0.1f;
        }
        //Right column bit
        tmp_image[ny*i+ny-1] = image[ny*i+ny-1] * 0.6f + image[ny*i-1] * 0.1f + image[ny*i+ny-2] * 0.1f + image[ny*i+2*ny-1] * 0.1f;
    }

	//Top rigtht corner
	tmp_image[ny*(nx-1)] = image[ny*(nx-1)] * 0.6f + image[ny*(nx-1)+1] * 0.1f + image[ny*(nx-2)] * 0.1f;

	//Right column
	for(int i = 1; i < ny-1; i++){
		tmp_image[ny*(nx-1)+i] = image[ny*(nx-1)+i] * 0.6f + image[ny*(nx-1)+i-1] * 0.1f + image[ny*(nx-1)+i-ny] * 0.1f + image[ny*(nx-1)+i+1] * 0.1f;
	}

	//Bottom right corner
	tmp_image[(nx*ny)-1] = image[(nx*ny)-1] * 0.6f + image[(nx*ny)-2] * 0.1f + image[(nx*ny)-1-ny] * 0.1f;


}


void stencilMaster(const int localCols, const int ny, const float *restrict image, float *restrict tmp_image){
  //__assume_aligned(image  );
  //__assume_aligned(tmp_image  );
  //__assume(localCols%16==0);
  //__assume(ny%16==0);
    //Top left corner
    tmp_image[0] = image[0] * 0.6f + image[1] * 0.1f + image[ny] * 0.1f;
    //Left Column
    for(int i = 1; i < ny-1; i++){
      tmp_image[i] = image[i] * 0.6f + image[i-1] * 0.1f + image[i+1] * 0.1f + image[i+ny] * 0.1f;
    }
    //Bottom left corner
    tmp_image[ny-1] = image[ny-1] * 0.6f + image[ny-2] * 0.1f + image[2*ny-1] * 0.1f;
    //Rest of columns
    for(int i = 1; i < localCols; i++){
      tmp_image[ny*i] = image[ny*i] * 0.6f + image[ny*i-ny] * 0.1f + image[ny*i+1] * 0.1f + image[ny*i+ny] * 0.1f;
      for(int j = 1; j < ny - 1; j++){
        tmp_image[j+i*ny] = image[j+i*ny] * 0.6f + image[j+i*ny-1] * 0.1f + image[j+i*ny+1] * 0.1f + image[j+i*ny-ny] * 0.1f + image[j + i*ny+ny] * 0.1f;
      }
      tmp_image[ny*i+ny-1] = image[ny*i+ny-1] * 0.6f + image[ny*i-1] * 0.1f + image[ny*i+ny-2] * 0.1f + image[ny*i+2*ny-1] * 0.1f;
    }
}

void stencilMidWorker(const int localCols, const int ny, const float *restrict image, float *restrict tmp_image){
  //__assume_aligned(image  );
  //__assume_aligned(tmp_image  );
  //__assume(localCols%16==0);
  //__assume(ny%16==0);
    //Rest of columns
    for(int i = 0; i < localCols; i++){
      tmp_image[ny*i] = image[ny*i] * 0.6f + image[ny*i-ny] * 0.1f + image[ny*i+1] * 0.1f + image[ny*i+ny] * 0.1f;
      for(int j = 1; j < ny - 1; j++){
        tmp_image[j+i*ny] = image[j+i*ny] * 0.6f + image[j+i*ny-1] * 0.1f + image[j+i*ny+1] * 0.1f + image[j+i*ny-ny] * 0.1f + image[j + i*ny+ny] * 0.1f;
      }
      tmp_image[ny*i+ny-1] = image[ny*i+ny-1] * 0.6f + image[ny*i-1] * 0.1f + image[ny*i+ny-2] * 0.1f + image[ny*i+2*ny-1] * 0.1f;
    }
}

void stencilFinWorker(const int localCols, const int ny, const float *restrict image, float *restrict tmp_image){
  //__assume_aligned(image  );
  //__assume_aligned(tmp_image  );
  //__assume(localCols%16==0);
  //__assume(ny%16==0);
    //Rest of columns

    for(int i = 0; i < localCols-1; i++){
      tmp_image[ny*i] = image[ny*i] * 0.6f + image[ny*i-ny] * 0.1f + image[ny*i+1] * 0.1f + image[ny*i+ny] * 0.1f;
      for(int j = 1; j < ny - 1; j++){
        tmp_image[j+i*ny] = image[j+i*ny] * 0.6f + image[j+i*ny-1] * 0.1f + image[j+i*ny+1] * 0.1f + image[j+i*ny-ny] * 0.1f + image[j + i*ny+ny] * 0.1f;
      }
      tmp_image[ny*i+ny-1] = image[ny*i+ny-1] * 0.6f + image[ny*i-1] * 0.1f + image[ny*i+ny-2] * 0.1f + image[ny*i+2*ny-1] * 0.1f;
    }
    //Top rigtht corner
  	tmp_image[ny*(localCols-1)] = image[ny*(localCols-1)] * 0.6f + image[ny*(localCols-1)+1] * 0.1f + image[ny*(localCols-2)] * 0.1f;
    //Right column
  	for(int i = 1; i < ny-1; i++){
  		tmp_image[ny*(localCols-1)+i] = image[ny*(localCols-1)+i] * 0.6f + image[ny*(localCols-1)+i-1] * 0.1f + image[ny*(localCols-1)+i-ny] * 0.1f + image[ny*(localCols-1)+i+1] * 0.1f;
  	}
    //Bottom right corner
  	tmp_image[(localCols*ny)-1] = image[(localCols*ny)-1] * 0.6f + image[(localCols*ny)-2] * 0.1f + image[(localCols*ny)-1-ny] * 0.1f;
}



//Create the input image
void init_image(const int nx, const int ny, float *  image, float *  tmp_image) {
  //Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0;
      tmp_image[j+i*ny] = 0.0;
    }
  }
  //Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2)
          image[jj+ii*ny] = 100.0;
        }
      }
    }
  }
}

//Routine to output the image in Netpbm grayscale binary image format
void output_image(const char * file_name, const int nx, const int ny, float *image) {

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  double maximum = 0.0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }
  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[j+i*ny]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}
