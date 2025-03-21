/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ROOFLINE_HH__
#define __ROOFLINE_HH__

namespace AstraSim {

class Roofline {
 public:
  Roofline(double peak_perf);
  Roofline(double bandwidth, double peak_perf);
  Roofline(double bandwidth, double peak_perf, double mfu);
  void set_bandwidth(double bandwidth);
  double get_perf(double operational_intensity, bool ignore_bandwidth);

 private:
  double bandwidth;
  double peak_perf;
  double mfu;
};

} // namespace AstraSim

#endif /* __ROOFLINE_HH__ */
