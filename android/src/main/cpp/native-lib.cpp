#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <string>
#include <vector>

#define LOG_TAG "RockEnhancerNative"
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

constexpr double kEps = 1e-8;

struct Image {
    int width{};
    int height{};
    std::vector<float> data; // Interleaved 3 channels.
};

inline float clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}

bool bitmapToImage(JNIEnv* env, jobject bitmap, Image& out) {
    AndroidBitmapInfo info{};
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        ALOGE("Failed to get bitmap info");
        return false;
    }

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        ALOGE("Bitmap format unsupported: %u", info.format);
        return false;
    }

    void* pixelsPtr = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixelsPtr) != ANDROID_BITMAP_RESULT_SUCCESS) {
        ALOGE("Failed to lock bitmap pixels");
        return false;
    }

    const uint8_t* pixels = static_cast<const uint8_t*>(pixelsPtr);
    out.width = static_cast<int>(info.width);
    out.height = static_cast<int>(info.height);
    out.data.resize(static_cast<size_t>(out.width) * static_cast<size_t>(out.height) * 3U);

    for (int y = 0; y < out.height; ++y) {
        const uint8_t* row = pixels + static_cast<size_t>(y) * info.stride;
        for (int x = 0; x < out.width; ++x) {
            const uint8_t* px = row + static_cast<size_t>(x) * 4U;
            const size_t idx = (static_cast<size_t>(y) * out.width + x) * 3U;
            out.data[idx + 0] = px[0] / 255.0f;
            out.data[idx + 1] = px[1] / 255.0f;
            out.data[idx + 2] = px[2] / 255.0f;
        }
    }

    AndroidBitmap_unlockPixels(env, bitmap);
    return true;
}

bool imageToBitmap(JNIEnv* env, const Image& image, jobject bitmap) {
    AndroidBitmapInfo info{};
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        ALOGE("Failed to get bitmap info on write");
        return false;
    }

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        info.width != static_cast<uint32_t>(image.width) ||
        info.height != static_cast<uint32_t>(image.height)) {
        ALOGE("Bitmap mismatch on write");
        return false;
    }

    void* pixelsPtr = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixelsPtr) != ANDROID_BITMAP_RESULT_SUCCESS) {
        ALOGE("Failed to lock bitmap for writing");
        return false;
    }

    uint8_t* pixels = static_cast<uint8_t*>(pixelsPtr);
    for (int y = 0; y < image.height; ++y) {
        uint8_t* row = pixels + static_cast<size_t>(y) * info.stride;
        for (int x = 0; x < image.width; ++x) {
            const size_t idx = (static_cast<size_t>(y) * image.width + x) * 3U;
            row[x * 4 + 0] = static_cast<uint8_t>(clamp01(image.data[idx + 0]) * 255.0f + 0.5f);
            row[x * 4 + 1] = static_cast<uint8_t>(clamp01(image.data[idx + 1]) * 255.0f + 0.5f);
            row[x * 4 + 2] = static_cast<uint8_t>(clamp01(image.data[idx + 2]) * 255.0f + 0.5f);
            row[x * 4 + 3] = 255;
        }
    }

    AndroidBitmap_unlockPixels(env, bitmap);
    return true;
}

Image resizeBilinear(const Image& src, int targetWidth) {
    if (targetWidth <= 0 || src.width <= 0 || src.height <= 0) {
        return src;
    }
    if (targetWidth == src.width) {
        return src;
    }

    const float scale = static_cast<float>(targetWidth) / static_cast<float>(src.width);
    const int targetHeight = std::max(1, static_cast<int>(std::round(src.height * scale)));

    Image out;
    out.width = targetWidth;
    out.height = targetHeight;
    out.data.resize(static_cast<size_t>(out.width) * static_cast<size_t>(out.height) * 3U);

    for (int y = 0; y < targetHeight; ++y) {
        const float gy = (static_cast<float>(y) + 0.5f) / scale - 0.5f;
        int y0 = static_cast<int>(std::floor(gy));
        int y1 = std::min(y0 + 1, src.height - 1);
        const float dy = gy - static_cast<float>(y0);
        y0 = std::max(y0, 0);

        for (int x = 0; x < targetWidth; ++x) {
            const float gx = (static_cast<float>(x) + 0.5f) / scale - 0.5f;
            int x0 = static_cast<int>(std::floor(gx));
            int x1 = std::min(x0 + 1, src.width - 1);
            const float dx = gx - static_cast<float>(x0);
            x0 = std::max(x0, 0);

            for (int c = 0; c < 3; ++c) {
                const float v00 = src.data[(static_cast<size_t>(y0) * src.width + x0) * 3U + c];
                const float v01 = src.data[(static_cast<size_t>(y0) * src.width + x1) * 3U + c];
                const float v10 = src.data[(static_cast<size_t>(y1) * src.width + x0) * 3U + c];
                const float v11 = src.data[(static_cast<size_t>(y1) * src.width + x1) * 3U + c];

                const float v0 = v00 + (v01 - v00) * dx;
                const float v1 = v10 + (v11 - v10) * dx;
                out.data[(static_cast<size_t>(y) * targetWidth + x) * 3U + c] = v0 + (v1 - v0) * dy;
            }
        }
    }

    return out;
}

void computeMeanAndCovariance(
    const Image& img,
    std::array<double, 3>& mean,
    double cov[3][3]
) {
    mean = {0.0, 0.0, 0.0};
    const size_t count = img.data.size() / 3U;
    if (count == 0) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                cov[r][c] = 0.0;
            }
        }
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        mean[0] += img.data[i * 3U + 0];
        mean[1] += img.data[i * 3U + 1];
        mean[2] += img.data[i * 3U + 2];
    }
    mean[0] /= static_cast<double>(count);
    mean[1] /= static_cast<double>(count);
    mean[2] /= static_cast<double>(count);

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            cov[r][c] = 0.0;
        }
    }

    for (size_t i = 0; i < count; ++i) {
        const double c0 = img.data[i * 3U + 0] - mean[0];
        const double c1 = img.data[i * 3U + 1] - mean[1];
        const double c2 = img.data[i * 3U + 2] - mean[2];

        cov[0][0] += c0 * c0;
        cov[0][1] += c0 * c1;
        cov[0][2] += c0 * c2;
        cov[1][0] += c1 * c0;
        cov[1][1] += c1 * c1;
        cov[1][2] += c1 * c2;
        cov[2][0] += c2 * c0;
        cov[2][1] += c2 * c1;
        cov[2][2] += c2 * c2;
    }

    const double denom = static_cast<double>(count > 1 ? (count - 1) : 1);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            cov[r][c] /= denom;
        }
    }
}

void jacobiEigenDecomposition(double a[3][3], double eigenValues[3], double eigenVectors[3][3]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            eigenVectors[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    constexpr int kMaxIter = 24;
    for (int iter = 0; iter < kMaxIter; ++iter) {
        int p = 0;
        int q = 1;
        double maxOff = std::fabs(a[0][1]);

        if (std::fabs(a[0][2]) > maxOff) {
            p = 0;
            q = 2;
            maxOff = std::fabs(a[0][2]);
        }
        if (std::fabs(a[1][2]) > maxOff) {
            p = 1;
            q = 2;
            maxOff = std::fabs(a[1][2]);
        }

        if (maxOff < 1e-12) {
            break;
        }

        const double app = a[p][p];
        const double aqq = a[q][q];
        const double apq = a[p][q];

        const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(phi);
        const double s = std::sin(phi);

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for (int k = 0; k < 3; ++k) {
            if (k != p && k != q) {
                const double akp = a[k][p];
                const double akq = a[k][q];
                a[k][p] = a[p][k] = c * akp - s * akq;
                a[k][q] = a[q][k] = s * akp + c * akq;
            }
        }

        for (int k = 0; k < 3; ++k) {
            const double vip = eigenVectors[k][p];
            const double viq = eigenVectors[k][q];
            eigenVectors[k][p] = c * vip - s * viq;
            eigenVectors[k][q] = s * vip + c * viq;
        }
    }

    eigenValues[0] = a[0][0];
    eigenValues[1] = a[1][1];
    eigenValues[2] = a[2][2];
}

void sortEigenPairsDescending(double eigenValues[3], double eigenVectors[3][3]) {
    for (int i = 0; i < 2; ++i) {
        int maxIdx = i;
        for (int j = i + 1; j < 3; ++j) {
            if (eigenValues[j] > eigenValues[maxIdx]) {
                maxIdx = j;
            }
        }
        if (maxIdx != i) {
            std::swap(eigenValues[i], eigenValues[maxIdx]);
            for (int r = 0; r < 3; ++r) {
                std::swap(eigenVectors[r][i], eigenVectors[r][maxIdx]);
            }
        }
    }
}

void pcaDecorrelateAndStretch(
    Image& img,
    double scale,
    const std::array<double, 3>* channelScales = nullptr
) {
    const size_t count = img.data.size() / 3U;
    if (count == 0) {
        return;
    }

    std::array<double, 3> mean{};
    double cov[3][3]{};
    computeMeanAndCovariance(img, mean, cov);

    double eigenValues[3]{};
    double eigenVectors[3][3]{};
    jacobiEigenDecomposition(cov, eigenValues, eigenVectors);
    sortEigenPairsDescending(eigenValues, eigenVectors);

    std::array<double, 3> projSum{0.0, 0.0, 0.0};
    std::array<double, 3> projSumSq{0.0, 0.0, 0.0};

    for (size_t i = 0; i < count; ++i) {
        const double c0 = img.data[i * 3U + 0] - mean[0];
        const double c1 = img.data[i * 3U + 1] - mean[1];
        const double c2 = img.data[i * 3U + 2] - mean[2];

        for (int k = 0; k < 3; ++k) {
            const double p =
                eigenVectors[0][k] * c0 +
                eigenVectors[1][k] * c1 +
                eigenVectors[2][k] * c2;
            projSum[k] += p;
            projSumSq[k] += p * p;
        }
    }

    std::array<double, 3> sigma{};
    const double n = static_cast<double>(count);
    for (int k = 0; k < 3; ++k) {
        const double mu = projSum[k] / n;
        const double var = std::max(0.0, projSumSq[k] / n - mu * mu);
        sigma[k] = std::sqrt(var);
        if (sigma[k] < kEps) {
            sigma[k] = 1.0;
        }
    }

    const double targetBase = (sigma[0] + sigma[1] + sigma[2]) / 3.0 * scale;
    std::array<double, 3> target{
        targetBase,
        targetBase,
        targetBase,
    };

    if (channelScales != nullptr) {
        for (int k = 0; k < 3; ++k) {
            target[k] *= (*channelScales)[k];
        }
    }

    for (size_t i = 0; i < count; ++i) {
        const double c0 = img.data[i * 3U + 0] - mean[0];
        const double c1 = img.data[i * 3U + 1] - mean[1];
        const double c2 = img.data[i * 3U + 2] - mean[2];

        double stretched[3]{};
        for (int k = 0; k < 3; ++k) {
            const double p =
                eigenVectors[0][k] * c0 +
                eigenVectors[1][k] * c1 +
                eigenVectors[2][k] * c2;
            stretched[k] = (p / sigma[k]) * target[k];
        }

        for (int j = 0; j < 3; ++j) {
            const double reconstructed =
                stretched[0] * eigenVectors[j][0] +
                stretched[1] * eigenVectors[j][1] +
                stretched[2] * eigenVectors[j][2] +
                mean[j];
            img.data[i * 3U + static_cast<size_t>(j)] = static_cast<float>(reconstructed);
        }
    }
}

Image allocateLike(const Image& src) {
    Image out;
    out.width = src.width;
    out.height = src.height;
    out.data.resize(src.data.size());
    return out;
}

double srgbToLinear(double c) {
    c = std::clamp(c, 0.0, 1.0);
    return c > 0.04045 ? std::pow((c + 0.055) / 1.055, 2.4) : c / 12.92;
}

double linearToSrgb(double c) {
    c = std::max(c, 0.0);
    return c > 0.0031308 ? (1.055 * std::pow(c, 1.0 / 2.4) - 0.055) : (12.92 * c);
}

double fLab(double t) {
    return t > 0.008856 ? std::cbrt(t) : (7.787 * t + 16.0 / 116.0);
}

double fLabInv(double t) {
    const double t3 = t * t * t;
    return t3 > 0.008856 ? t3 : (t - 16.0 / 116.0) / 7.787;
}

void rgbToYxx(const Image& rgb, Image& yxx, double yMul, double uMul, double vMul) {
    yxx = allocateLike(rgb);
    const size_t count = rgb.data.size() / 3U;
    for (size_t i = 0; i < count; ++i) {
        const double r = rgb.data[i * 3U + 0];
        const double g = rgb.data[i * 3U + 1];
        const double b = rgb.data[i * 3U + 2];

        const double y = 0.299 * r + 0.587 * g + 0.114 * b;
        const double u = yMul * (b - uMul * y);
        const double v = yMul * (r - vMul * y);

        yxx.data[i * 3U + 0] = static_cast<float>(y);
        yxx.data[i * 3U + 1] = static_cast<float>(u);
        yxx.data[i * 3U + 2] = static_cast<float>(v);
    }
}

void yxxToRgb(const Image& yxx, Image& rgb, double yMul, double uMul, double vMul) {
    rgb = allocateLike(yxx);
    const double safeYMul = std::fabs(yMul) < kEps ? 1.0 : yMul;
    const size_t count = yxx.data.size() / 3U;

    for (size_t i = 0; i < count; ++i) {
        const double y = yxx.data[i * 3U + 0];
        const double u = yxx.data[i * 3U + 1];
        const double v = yxx.data[i * 3U + 2];

        const double r = v / safeYMul + vMul * y;
        const double b = u / safeYMul + uMul * y;
        const double g = (y - 0.299 * r - 0.114 * b) / 0.587;

        rgb.data[i * 3U + 0] = static_cast<float>(r);
        rgb.data[i * 3U + 1] = static_cast<float>(g);
        rgb.data[i * 3U + 2] = static_cast<float>(b);
    }
}

void rgbToLxx(
    const Image& rgb,
    Image& lxx,
    double lxxmul1,
    double lxxmul2,
    double lxxmula,
    double lxxmulb
) {
    constexpr double d65Xn = 95.047;
    constexpr double d65Yn = 100.0;
    constexpr double d65Zn = 108.883;

    lxx = allocateLike(rgb);
    const size_t count = rgb.data.size() / 3U;

    const double safeMul1 = std::fabs(lxxmul1) < kEps ? 1.0 : lxxmul1;
    const double safeMul2 = std::fabs(lxxmul2) < kEps ? 1.0 : lxxmul2;

    for (size_t i = 0; i < count; ++i) {
        const double r = srgbToLinear(rgb.data[i * 3U + 0]);
        const double g = srgbToLinear(rgb.data[i * 3U + 1]);
        const double b = srgbToLinear(rgb.data[i * 3U + 2]);

        const double x = (0.4124 * r + 0.3576 * g + 0.1805 * b) * 100.0;
        const double y = (0.2126 * r + 0.7152 * g + 0.0722 * b) * 100.0;
        const double z = (0.0193 * r + 0.1192 * g + 0.9505 * b) * 100.0;

        const double fx = fLab(x / d65Xn);
        const double fy = fLab(y / d65Yn);
        const double fz = fLab(z / d65Zn);

        const double l = 116.0 * fy - 16.0;
        const double a = (1.0 / safeMul1) * 250.0 * (fx - lxxmula * fy);
        const double bb = (1.0 / safeMul2) * 100.0 * (lxxmulb * fy - fz);

        lxx.data[i * 3U + 0] = static_cast<float>(l);
        lxx.data[i * 3U + 1] = static_cast<float>(a);
        lxx.data[i * 3U + 2] = static_cast<float>(bb);
    }
}

void lxxToRgb(
    const Image& lxx,
    Image& rgb,
    double lxxmul1,
    double lxxmul2,
    double lxxmula,
    double lxxmulb
) {
    constexpr double d65Xn = 95.047;
    constexpr double d65Yn = 100.0;
    constexpr double d65Zn = 108.883;

    rgb = allocateLike(lxx);
    const size_t count = lxx.data.size() / 3U;

    for (size_t i = 0; i < count; ++i) {
        const double l = lxx.data[i * 3U + 0];
        const double a = lxx.data[i * 3U + 1];
        const double b = lxx.data[i * 3U + 2];

        const double fy = (l + 16.0) / 116.0;
        const double fx = lxxmul1 * a * 0.004 + lxxmula * fy;
        const double fz = fy * lxxmulb - lxxmul2 * b * 0.01;

        const double x = fLabInv(fx) * d65Xn / 100.0;
        const double y = fLabInv(fy) * d65Yn / 100.0;
        const double z = fLabInv(fz) * d65Zn / 100.0;

        const double rl = 3.2406 * x + -1.5372 * y + -0.4986 * z;
        const double gl = -0.9689 * x + 1.8758 * y + 0.0415 * z;
        const double bl = 0.0557 * x + -0.2040 * y + 1.0570 * z;

        rgb.data[i * 3U + 0] = static_cast<float>(linearToSrgb(rl));
        rgb.data[i * 3U + 1] = static_cast<float>(linearToSrgb(gl));
        rgb.data[i * 3U + 2] = static_cast<float>(linearToSrgb(bl));
    }
}

void clipImage(Image& img) {
    for (float& v : img.data) {
        v = clamp01(v);
    }
}

Image applyYxxPcaStretch(
    const Image& rgb,
    double scale,
    const std::array<double, 3>& yxxScales
) {
    Image yxx;
    rgbToYxx(rgb, yxx, yxxScales[0], yxxScales[1], yxxScales[2]);
    pcaDecorrelateAndStretch(yxx, scale);

    Image out;
    yxxToRgb(yxx, out, yxxScales[0], yxxScales[1], yxxScales[2]);
    clipImage(out);
    return out;
}

Image applyLxxPcaStretch(
    const Image& rgb,
    double scale,
    const std::array<double, 4>& lxxScales
) {
    Image lxx;
    rgbToLxx(rgb, lxx, lxxScales[0], lxxScales[1], lxxScales[2], lxxScales[3]);
    pcaDecorrelateAndStretch(lxx, scale);

    Image out;
    lxxToRgb(lxx, out, lxxScales[0], lxxScales[1], lxxScales[2], lxxScales[3]);
    clipImage(out);
    return out;
}

std::string toLowerAscii(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::vector<float> readFactors(JNIEnv* env, jfloatArray factors) {
    if (factors == nullptr) {
        return {};
    }
    const jsize len = env->GetArrayLength(factors);
    if (len <= 0) {
        return {};
    }
    std::vector<float> out(static_cast<size_t>(len));
    env->GetFloatArrayRegion(factors, 0, len, out.data());
    return out;
}

} // namespace

extern "C" JNIEXPORT jint JNICALL
Java_expo_modules_rockenhancer_RockenhancerModule_nativeProcess(
    JNIEnv* env,
    jobject /*thiz*/,
    jstring jInputPath,
    jstring jOutputPath,
    jint targetWidth,
    jint jpegQuality,
    jstring jMode,
    jfloatArray factors
) {
    if (jInputPath == nullptr || jOutputPath == nullptr) {
        ALOGE("Input/output path is null");
        return -10;
    }

    const char* inPathChars = env->GetStringUTFChars(jInputPath, nullptr);
    const char* outPathChars = env->GetStringUTFChars(jOutputPath, nullptr);
    if (inPathChars == nullptr || outPathChars == nullptr) {
        if (inPathChars != nullptr) {
            env->ReleaseStringUTFChars(jInputPath, inPathChars);
        }
        if (outPathChars != nullptr) {
            env->ReleaseStringUTFChars(jOutputPath, outPathChars);
        }
        ALOGE("Failed to read input/output path chars");
        return -11;
    }

    std::string inputPath(inPathChars);
    std::string outputPath(outPathChars);
    env->ReleaseStringUTFChars(jInputPath, inPathChars);
    env->ReleaseStringUTFChars(jOutputPath, outPathChars);

    std::string mode = "yxx";
    if (jMode != nullptr) {
        const char* modeChars = env->GetStringUTFChars(jMode, nullptr);
        if (modeChars != nullptr) {
            mode = toLowerAscii(std::string(modeChars));
            env->ReleaseStringUTFChars(jMode, modeChars);
        }
    }

    const std::vector<float> factorsVec = readFactors(env, factors);

    jclass optionsCls = env->FindClass("android/graphics/BitmapFactory$Options");
    jmethodID optionsCtor = env->GetMethodID(optionsCls, "<init>", "()V");
    jobject optionsObj = env->NewObject(optionsCls, optionsCtor);

    jfieldID inPreferredConfigField = env->GetFieldID(
        optionsCls,
        "inPreferredConfig",
        "Landroid/graphics/Bitmap$Config;"
    );

    jclass bitmapConfigCls = env->FindClass("android/graphics/Bitmap$Config");
    jmethodID valueOfConfig = env->GetStaticMethodID(
        bitmapConfigCls,
        "valueOf",
        "(Ljava/lang/String;)Landroid/graphics/Bitmap$Config;"
    );
    jstring argbStr = env->NewStringUTF("ARGB_8888");
    jobject argbConfig = env->CallStaticObjectMethod(bitmapConfigCls, valueOfConfig, argbStr);
    env->SetObjectField(optionsObj, inPreferredConfigField, argbConfig);
    env->DeleteLocalRef(argbStr);

    jclass bitmapFactoryCls = env->FindClass("android/graphics/BitmapFactory");
    jmethodID decodeMethod = env->GetStaticMethodID(
        bitmapFactoryCls,
        "decodeFile",
        "(Ljava/lang/String;Landroid/graphics/BitmapFactory$Options;)Landroid/graphics/Bitmap;"
    );

    jstring inputPathJ = env->NewStringUTF(inputPath.c_str());
    jobject bitmap = env->CallStaticObjectMethod(bitmapFactoryCls, decodeMethod, inputPathJ, optionsObj);
    env->DeleteLocalRef(inputPathJ);

    if (bitmap == nullptr) {
        ALOGE("Failed to decode bitmap at %s", inputPath.c_str());
        return -1;
    }

    Image image{};
    if (!bitmapToImage(env, bitmap, image)) {
        env->DeleteLocalRef(bitmap);
        return -2;
    }

    const int desiredWidth = targetWidth > 0 ? targetWidth : 1980;
    const int safeJpegQuality = (jpegQuality >= 1 && jpegQuality <= 100) ? jpegQuality : 50;
    constexpr double kDefaultScale = 1.0;

    Image resized = resizeBilinear(image, desiredWidth);
    Image finalImg;

    if (mode == "lxx") {
        std::array<double, 4> lxxScales{1.0, 1.0, 1.0, 1.0};
        if (factorsVec.size() >= 4) {
            lxxScales = {
                factorsVec[0],
                factorsVec[1],
                factorsVec[2],
                factorsVec[3],
            };
        } else if (factorsVec.size() >= 3) {
            lxxScales = {
                factorsVec[0],
                factorsVec[0],
                factorsVec[1],
                factorsVec[2],
            };
        }
        finalImg = applyLxxPcaStretch(resized, kDefaultScale, lxxScales);
    } else {
        std::array<double, 3> yxxScales{1.0, 0.8, 0.4};
        if (factorsVec.size() >= 3) {
            yxxScales = {
                factorsVec[0],
                factorsVec[1],
                factorsVec[2],
            };
        }
        finalImg = applyYxxPcaStretch(resized, kDefaultScale, yxxScales);
    }

    jclass bitmapCls = env->FindClass("android/graphics/Bitmap");
    jmethodID createBitmapMethod = env->GetStaticMethodID(
        bitmapCls,
        "createBitmap",
        "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;"
    );

    jobject outBitmap = env->CallStaticObjectMethod(
        bitmapCls,
        createBitmapMethod,
        finalImg.width,
        finalImg.height,
        argbConfig
    );

    if (outBitmap == nullptr) {
        ALOGE("Failed to allocate output bitmap");
        env->DeleteLocalRef(bitmap);
        return -3;
    }

    if (!imageToBitmap(env, finalImg, outBitmap)) {
        env->DeleteLocalRef(bitmap);
        env->DeleteLocalRef(outBitmap);
        return -4;
    }

    jclass fosCls = env->FindClass("java/io/FileOutputStream");
    jmethodID fosCtor = env->GetMethodID(fosCls, "<init>", "(Ljava/lang/String;)V");
    jstring outputPathJ = env->NewStringUTF(outputPath.c_str());
    jobject fos = env->NewObject(fosCls, fosCtor, outputPathJ);
    env->DeleteLocalRef(outputPathJ);

    if (fos == nullptr) {
        ALOGE("Failed to create FileOutputStream for %s", outputPath.c_str());
        env->DeleteLocalRef(bitmap);
        env->DeleteLocalRef(outBitmap);
        return -5;
    }

    jclass compressFormatCls = env->FindClass("android/graphics/Bitmap$CompressFormat");
    jfieldID jpegField = env->GetStaticFieldID(
        compressFormatCls,
        "JPEG",
        "Landroid/graphics/Bitmap$CompressFormat;"
    );
    jobject jpegFormat = env->GetStaticObjectField(compressFormatCls, jpegField);

    jmethodID compressMethod = env->GetMethodID(
        bitmapCls,
        "compress",
        "(Landroid/graphics/Bitmap$CompressFormat;ILjava/io/OutputStream;)Z"
    );

    const jboolean compressed = env->CallBooleanMethod(outBitmap, compressMethod, jpegFormat, safeJpegQuality, fos);

    jmethodID closeMethod = env->GetMethodID(fosCls, "close", "()V");
    env->CallVoidMethod(fos, closeMethod);

    env->DeleteLocalRef(bitmap);
    env->DeleteLocalRef(outBitmap);
    env->DeleteLocalRef(fos);

    return compressed ? 0 : -6;
}
