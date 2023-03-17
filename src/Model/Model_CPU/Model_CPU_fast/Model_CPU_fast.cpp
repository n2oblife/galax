#define GALAX_MODEL_CPU_FAST
#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
using b_type = xs::batch<float>;

Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_fast
::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

    std::vector<float> distances(particles.x.size(), 0.0);

    // OMP  version
    #pragma omp parallel for
        for (int i = 0; i < n_particles; i++)
        {
            for (int j = 0; j < n_particles; j++) // we avoid computing two times the same operation
            {
                if(i!=j)
                {
                    const float diffx = particles.x[j] - particles.x[i];
                    const float diffy = particles.y[j] - particles.y[i];
                    const float diffz = particles.z[j] - particles.z[i];

                    float dij = diffx * diffx + diffy * diffy + diffz * diffz;

                    if (dij < 1.0)
                    {
                        dij = 10.0;
                    }
                    else
                    {
                        dij = std::sqrt(dij);
                        dij = 10.0 / (dij * dij * dij);
                    }

                    accelerationsx[i] += diffx * dij * initstate.masses[j];
                    accelerationsy[i] += diffy * dij * initstate.masses[j];
                    accelerationsz[i] += diffz * dij * initstate.masses[j];
                }
            }
        }


    // OMP + xsimd version
    #pragma omp parallel for

    // Chargement des données pour éviter que ca arrive dans la boucle
    b_type two = 2;
    b_type tot = 3/2;
    b_type ten = 10; // penser à convertir tous les floats et regarder si il y a amélioration en les créant ou sortant de la boucle

    for (int i = 0; i < n_particles; i += b_type::size)
    {
        // load registers body i
        const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);  

        for(int j=0; j < n_particles; j++)
        {   
            if(i!=j)
            {
                b_type diffx = xs::sub(particles.x[j],particles.x[i]);
                b_type diffy = xs::sub(particles.y[j],particles.y[i]);
                b_type diffz = xs::sub(particles.z[j],particles.z[i]);
                
                b_type dij = xs::add(xs::add(xs::pow(diffx,two),xs::pow(diffy,two)),xs::pow(diffz,two));
                dij = xs::min(ten,xs::div(ten,xs::pow(dij,tot))); // comparison to 1

                b_type massej = b_type::load_unaligned(&initstate.masses[j]);

                raccx_i = xs::add(raccx_i,xs::mul(diffx,xs::mul(dij,massej)));
                raccy_i = xs::add(raccy_i,xs::mul(diffy,xs::mul(dij,massej)));
                raccz_i = xs::add(raccz_i,xs::mul(diffz,xs::mul(dij,massej)));
            }
        }  
    }

}

#endif // GALAX_MODEL_CPU_FAST
