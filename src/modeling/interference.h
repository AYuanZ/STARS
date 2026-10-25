#ifndef STARS_INTERFERENCE_H
#define STARS_INTERFERENCE_H

/* Empirical coefficients copied from the original scheduler snapshot. */
double stars_predict_interference(double dram, double l2, int colocated_sms,
                                 double colocated_dram, double colocated_l2);
double stars_predict_sm_duration(double known_sms, int new_sms, double duration);
#endif
