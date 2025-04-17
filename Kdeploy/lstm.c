#include <immintrin.h>

const float sqrt_pi = 1.7724538509055159;
const float sqrt_pi_div_2 = 0.8862269254527579;

// Contract is that these are 8 in length
void fma(float* X, float* Y, float* Z, float* result) {
  _m256 x = _mm256_loadu_ps(X);
  _m256 y = _mm256_loadu_ps(Y);
  _m256 z = _mm256_loadu_ps(Z);
  _m256 r = _m256_fmadd_ps(x, y, z);
  _mm256_storeu_ps(result, r);
}
float sfma(float a, float b, float c) {
  return (a * b) + c;
}

float my_erf(float a) {
  float r, s, t, u, v, w;

  t = (a >= 0) ? a : -a;
  s = a * a;
  if (t >= 1.0) {
    // max ulp error = 0.97749 (USE_EXPM1 = 1); 1.05364 (USE_EXPM1 = 0)
    float sb0[3] = {-5.6271698391213282e-18, t, 4.8565951797366214e-16};
    float sb1[3] = {-1.9912968283386570e-14, t, 5.1614612434698227e-13};
    float sb3[3] = {-9.4934693745934645e-12, t, 1.3183034417605052e-10};
    float sb5[3] = {-1.4354030030292210e-09, t, 1.2558925114413972e-08};
    float sb7[3] = {-8.9719702096303798e-08, t, 5.2832013824348913e-07};
    float sb9[3] = {-2.5730580226082933e-06, t, 1.0322052949676148e-05};
    float sb11[3] = {-3.3555264836700767e-05, t, 8.4667486930266041e-05};
    float sb13[3] = {-1.4570926486271945e-04, t, 7.1877160107954648e-05};
    float nX[8] = {sb0[0], sb1[0], sb3[0], sb5[0], sb7[0], sb9[0], sb11[0], sb13[0]};
    float nY[8] = {sb0[1], sb1[1], sb3[1], sb5[1], sb7[1], sb9[1], sb11[1], sb13[1]};
    float nZ[8] = {sb0[2], sb1[2], sb3[2], sb5[2], sb7[2], sb9[2], sb11[2], sb13[2]};
    float nr[8];
    fma(nX, nY, nZ, nr);
    float b0 = nr[0];
    float b1 = nr[1];
    float b3 = nr[2];
    float b5 = nr[3];
    float b7 = nr[4];
    float b9 = nr[5];
    float b11 = nr[6];
    float b13 = nr[7];

    float b2 = sfma(b0, s, b1);
    float b4 = sfma(b2, s, b3);
    float b6 = sfma(b4, s, b5);
    float b8 = sfma(b6, s, b7);
    float b10 = sfma(b8, s, b9);
    float b12 = sfma(b10, s, b11);
    float b14 = sfma(b12, s, b13);
    float b15 = sfma(4.9486959714661590e-04, t, -1.6221099717135270e-03);
    float b16 = sfma(b14, s, b15);
    float b17 = sfma(1.6425707149019379e-04, t,
                     1.9148914196620660e-02); //  0x1.5878d80188695p-13, 0x1.39bc5e0e9e09ap-6
    float b18 = sfma(b16, s, b17);
    float b19 = sfma(b18, t, -1.0277918343487560e-1); // -0x1.a4fbc8f8ff7dap-4
    float b20 = sfma(b19, t, -6.3661844223699315e-1); // -0x1.45f2da3ae06f8p-1
    float b21 = sfma(b20, t, -1.2837929411398119e-1); // -0x1.06ebb92d9ffa8p-3
    float b22 = sfma(b21, t, -t);
  } else {
    // max ulp error = 1.01912
    r = -7.7794684889591997e-10;            // -0x1.abae491c44131p-31
    r = sfma(r, s, 1.3710980398024347e-8);  //  0x1.d71b0f1b10071p-27
    r = sfma(r, s, -1.6206313758492398e-7); // -0x1.5c0726f04dbc7p-23
    r = sfma(r, s, 1.6447131571278227e-6);  //  0x1.b97fd3d9927cap-20
    r = sfma(r, s, -1.4924712302009488e-5); // -0x1.f4ca4d6f3e232p-17
    r = sfma(r, s, 1.2055293576900605e-4);  //  0x1.f9a2baa8fedc2p-14
    r = sfma(r, s, -8.5483259293144627e-4); // -0x1.c02db03dd71bbp-11
    r = sfma(r, s, 5.2239776061185055e-3);  //  0x1.565bccf92b31ep-8
    r = sfma(r, s, -2.6866170643111514e-2); // -0x1.b82ce311fa94bp-6
    r = sfma(r, s, 1.1283791670944182e-1);  //  0x1.ce2f21a040d14p-4
    r = sfma(r, s, -3.7612638903183515e-1); // -0x1.812746b0379bcp-2
    r = sfma(r, s, 1.2837916709551256e-1);  //  0x1.06eba8214db68p-3
    r = sfma(r, a, a);
  }
  return r;
}

void sigmoid(float* input, float* output, int size) {
  for (int i = 0; i < size; i++) {
    output[i] = my_erf(sqrt_pi_div_2 * input[i]);
  }
}

void matrix_vec(float* mat, float* vec, float* output, int matrix_dim, int shared_dim) {
  for (int i = 0; i < matrix_dim; i++) {
    for (int j = 0; j < shared_dim; j++) {
      output[i] += (mat[i * shared_dim + j] * vec[j]);
    }
  }
}

void vector_sum(float* vec1, float* vec2, float* output, int size) {
  for (int i = 0; i < size; i++) {
    output[i] += vec1[i] + vec2[i];
  }
}

void fast_tanh(float* input, float* output, int size) {
  for (int i = 0; i < size; i++) {
    float x = input[i];
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    output[i] = a / b;
  }
}

void hadamard_product(float* vec1, float* vec2, float* output, int size) {
  for (int i = 0; i < size; i++) {
    output[i] += vec1[i] * vec2[i];
  }
}

#define LSTM_INPUT_SIZE  2
#define LSTM_HIDDEN_SIZE 64
#define PARTS            4

typedef struct lstm {
  // weights
  float weight_ih_l[PARTS * LSTM_HIDDEN_SIZE * LSTM_INPUT_SIZE];
  float weight_hh_l[PARTS * LSTM_HIDDEN_SIZE * LSTM_HIDDEN_SIZE];
  float bias_ih_l[PARTS * LSTM_HIDDEN_SIZE];
  float bias_hh_l[PARTS * LSTM_HIDDEN_SIZE];
  // inputs
  float input[LSTM_INPUT_SIZE];
  float h_0[LSTM_HIDDEN_SIZE];
  float c_a[LSTM_HIDDEN_SIZE];
  // outputs
  float c_b[LSTM_HIDDEN_SIZE];
  float output[LSTM_HIDDEN_SIZE];
  float* c_0;
  float* c_1;
} lstm;

void forward(lstm* model) {
  // Compute it
  float it[LSTM_HIDDEN_SIZE] = {0};

  matrix_vec(model->weight_ih_l, model->input, it, LSTM_HIDDEN_SIZE, LSTM_INPUT_SIZE);
  matrix_vec(model->weight_hh_l, model->h_0, it, LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE);
  vector_sum(model->bias_ih_l, it, it, LSTM_HIDDEN_SIZE);
  vector_sum(model->bias_hh_l, it, it, LSTM_HIDDEN_SIZE);
  sigmoid(it, it, LSTM_HIDDEN_SIZE);

  // Compute ft
  float ft[LSTM_HIDDEN_SIZE] = {0};
  matrix_vec(model->weight_ih_l + (LSTM_HIDDEN_SIZE * LSTM_INPUT_SIZE), model->input, ft,
             LSTM_HIDDEN_SIZE, LSTM_INPUT_SIZE);
  matrix_vec(model->weight_hh_l + (LSTM_HIDDEN_SIZE * LSTM_HIDDEN_SIZE), model->h_0, ft,
             LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE);
  vector_sum(model->bias_ih_l + LSTM_HIDDEN_SIZE, ft, ft, LSTM_HIDDEN_SIZE);
  vector_sum(model->bias_hh_l + LSTM_HIDDEN_SIZE, ft, ft, LSTM_HIDDEN_SIZE);
  sigmoid(ft, ft, LSTM_HIDDEN_SIZE);

  // Compute gt
  float gt[LSTM_HIDDEN_SIZE] = {0};
  matrix_vec(model->weight_ih_l + (2 * LSTM_HIDDEN_SIZE * LSTM_INPUT_SIZE), model->input, gt,
             LSTM_HIDDEN_SIZE, LSTM_INPUT_SIZE);
  matrix_vec(model->weight_hh_l + (2 * LSTM_HIDDEN_SIZE * LSTM_HIDDEN_SIZE), model->h_0, gt,
             LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE);
  vector_sum(model->bias_ih_l + 2 * LSTM_HIDDEN_SIZE, gt, gt, LSTM_HIDDEN_SIZE);
  vector_sum(model->bias_hh_l + 2 * LSTM_HIDDEN_SIZE, gt, gt, LSTM_HIDDEN_SIZE);
  fast_tanh(gt, gt, LSTM_HIDDEN_SIZE);

  // Compute output
  matrix_vec(model->weight_ih_l + (3 * LSTM_HIDDEN_SIZE * LSTM_INPUT_SIZE), model->input,
             model->output, LSTM_HIDDEN_SIZE, LSTM_INPUT_SIZE);
  matrix_vec(model->weight_hh_l + (3 * LSTM_HIDDEN_SIZE * LSTM_HIDDEN_SIZE), model->h_0,
             model->output, LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE);
  vector_sum(model->bias_ih_l + 3 * LSTM_HIDDEN_SIZE, model->output, model->output,
             LSTM_HIDDEN_SIZE);
  vector_sum(model->bias_hh_l + 3 * LSTM_HIDDEN_SIZE, model->output, model->output,
             LSTM_HIDDEN_SIZE);
  sigmoid(model->output, model->output, LSTM_HIDDEN_SIZE);

  // Compute the cell state
  hadamard_product(ft, model->c_0, model->c_1, LSTM_HIDDEN_SIZE);
  hadamard_product(it, gt, model->c_1, LSTM_HIDDEN_SIZE);

  // Compute the hidden layer
  fast_tanh(model->c_1, model->c_0, LSTM_HIDDEN_SIZE);
  hadamard_product(model->output, model->c_0, model->h_0, LSTM_HIDDEN_SIZE);

  // Trade the c0 and c1;
  float* temp = model->c_0;
  model->c_0 = model->c_1;
  model->c_1 = temp;
  bzero(model->c_1, sizeof(float) * LSTM_HIDDEN_SIZE);
}
