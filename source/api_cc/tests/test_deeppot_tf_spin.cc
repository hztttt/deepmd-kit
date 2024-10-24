// SPDX-License-Identifier: LGPL-3.0-or-later
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "DeepPot.h"
#include "neighbor_list.h"
#include "test_utils.h"

template <class VALUETYPE>
class TestInferDeepPotSpin : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81};
  std::vector<VALUETYPE> spin = {0., 0., 1.2737, 0., 0., 1.2737,
                                 0., 0., 0., 0., 0., 0.};
  std::vector<int> atype = {0, 0, 1, 1, 2, 2};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<VALUETYPE> expected_e = {
      -7.364038132599981 , -7.3636699370168355,
      -4.1290542830731045, -4.129532531949317};
  std::vector<VALUETYPE> expected_f = {
      -0.037830295809031,  0.0154057392094471,  0.0261077517781748,
      0.0373293046333845, -0.0153743595168964, -0.0263245484923275,
      -0.0211205732587445, -0.0206061123926398,  0.0302326903161227,
      0.021621564434391 ,  0.0205747327000891, -0.0300158936019702};
  std::vector<VALUETYPE> expected_fm = {
      -0.007727364895312 ,  0.0031995060137736, -0.0809320545918724,
      0.0068146686388984, -0.0027772649187765, -0.0793843059157293,
      0.0000000000000000, 0.00000000000000000, 0.00000000000000000,
      0.0000000000000000, 0.00000000000000000, 0.00000000000000000};
  int natoms;
  double expected_tot_e;

  deepmd::DeepPot dp;

  void SetUp() override {
    std::string file_name = "../../tests/infer/deepspin.pbtxt";
    deepmd::convert_pbtxt_to_pb("../../tests/infer/deepspin.pbtxt",
                                "deepspin.pb");

    dp.init("deepspin.pb");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 3, expected_fm.size());
    expected_tot_e = 0.;
    for (int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
  };

  void TearDown() override { remove("deeppot.pb"); };
};

TYPED_TEST_SUITE(TestInferDeepPotSpin, ValueTypes);

TYPED_TEST(TestInferDeepPotSpin, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  dp.compute(ener, force, force_mag, virial, coord, spin, atype, box);
  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
}

TYPED_TEST(TestInferDeepPotSpin, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial, atom_ener, atom_vir;
  dp.compute(ener, force, force_mag, virial, atom_ener, atom_vir, coord, spin,
             atype, box);
  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  EXPECT_EQ(atom_ener.size(), natoms);
  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), EPSILON);
  }
}
