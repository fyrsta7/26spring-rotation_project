/**
 * @file
 * linear least squares model
 */

#include <math.h>
#include <string.h>

#include "attributes.h"
#include "version.h"
#include "lls.h"

static void update_lls(LLSModel *m, const double *var)
{
    int i, j;

    for (i = 0; i <= m->indep_count; i++) {
        for (j = i; j <= m->indep_count; j++) {
            m->covariance[i][j] += var[i] * var[j];
        }
    }
}

void avpriv_solve_lls(LLSModel *m, double threshold, unsigned short min_order)
{
    int i, j, k;
    double (*factor)[MAX_VARS_ALIGN] = (void *) &m->covariance[1][0];
    double (*covar) [MAX_VARS_ALIGN] = (void *) &m->covariance[1][1];
    double *covar_y                = m->covariance[0];
    int count                      = m->indep_count;

    for (i = 0; i < count; i++) {
        for (j = i; j < count; j++) {
            double sum = covar[i][j];

            for (k = i - 1; k >= 0; k--)
                sum -= factor[i][k] * factor[j][k];

            if (i == j) {
                if (sum < threshold)
                    sum = 1.0;
                factor[i][i] = sqrt(sum);
            } else {
                factor[j][i] = sum / factor[i][i];
            }
        }
    }

    for (i = 0; i < count; i++) {
        double sum = covar_y[i + 1];

        for (k = i - 1; k >= 0; k--)
            sum -= factor[i][k] * m->coeff[0][k];

        m->coeff[0][i] = sum / factor[i][i];
    }
