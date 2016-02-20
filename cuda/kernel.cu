/*--------------------------In order to use printf ---------*/
extern "C" {
   #include <stdio.h>
}

/* -----------------------------------------------------------------------------
                Declare Textures and Constant Memory
----------------------------------------------------------------------------- */

{% for tex in textures %}texture<unsigned char, 2> t_ucFitnessFunctionGrids{{ tex }};
{% endfor %}
texture<float, 2> t_ucInteractionMatrix; 

__constant__ float c_fParams[{{ params_size }}];
__constant__ float c_fFitnessParams[{{ fitnessparams_size }}];
__constant__ unsigned char c_ucFourPermutations[{{ fit_nr_fourpermutations }}][{{ glob_nr_tileorientations }}];
__constant__ float c_fFitnessSumConst;
__constant__ float c_fFitnessListConst[{{ glob_nr_genomes }}];
__constant__ float c_fGAParams[{{ gaparams_size }}];

/* -----------------------------------------------------------------------------
                Include All Header Files
----------------------------------------------------------------------------- */
#include "curand_kernel.h"

extern "C"
{

//start include globals.inc.cuh
{% include 'globals.inc.cuh' %}
//end include globals.inc.cuh

//start include curandinit.inc.cuh
{% include 'curandinit.inc.cuh' %}
//end include curandinit.inc.cuh

//start include fitness.inc.cuh
{#{% include 'fitness.inc.cuh' %}#}
//end include fitness.inc.cuh

//start include sorting.inc.cuh
{#{% include 'sorting.inc.cuh' %}#}
//end include sorting.inc.cuh

//start include ga_utils.inc.cuh
{#{% include 'ga_utils.inc.cuh' %}#}
//end include ga_utils.inc.cuh

//start include ga.inc.cuh
{#{% include 'ga.inc.cuh' %}#}
//end include ga.inc.cuh

/* -----------------------------------------------------------------------------
                Define Kernels
----------------------------------------------------------------------------- */
    
    //start include globals.inc.cu
{% include 'globals.inc.cu' %}
    //end include globals.inc.cu

    //start include curandinit.inc.cu
{% include 'curandinit.inc.cu' %}
    //end include curandinit.inc.cu

    //start include fitness.inc.cu
{% include 'fitness.inc.cu' %}
    //end include fitness.inc.cu

    //start include sorting.inc.cu
{#{% include 'sorting.inc.cu' %}#}
    //end include sorting.inc.cu

    //start include ga_utils.inc.cu
{#{% include 'ga_utils.inc.cu' %}#}
    //end include ga_utils.inc.cu   

    //start include ga.inc.cu
{#{% include 'ga.inc.cu' %}#}
    //end include ga.inc.cu    
}
