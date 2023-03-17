#ifndef DIFF_HPP_
#define DIFF_HPP_

#include <string>
#include <vector>

class Diff{
private :
    const int max_n_particles  = 81920;

public:
    std::vector<float> diffx;
    std::vector<float> diffy;
    std::vector<float> diffz;

    Diff(const int n_particles)
};

#endif // DIFF_HPP_