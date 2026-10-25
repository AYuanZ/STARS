#include <math.h>
#include "interference.h"

double stars_predict_interference(double dram, double l2, int colocated_sms,
                                 double colocated_dram, double colocated_l2)
{
    return dram * 6.86 + l2 * 9.29 + colocated_sms * 24.64
           + colocated_dram * 10.16 + colocated_l2 * 9.73;
}

double stars_predict_sm_duration(double known_sms, int new_sms, double duration)
{
    return duration * pow(new_sms / known_sms, -0.768);
}
