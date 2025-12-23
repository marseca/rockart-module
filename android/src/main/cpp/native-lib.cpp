#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>

#define LOG_TAG "RockEnhancerNative"
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct Image {
    int width{};
    int height{};
    std::vector<float> data; // RGB, normalized 0..1
};

inline float clamp01(float v) {
    return std::fmax(0.f, std::fmin(1.f, v));
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

    const uint8_t* pixels = static_cast<uint8_t*>(pixelsPtr);
    out.width = static_cast<int>(info.width);
    out.height = static_cast<int>(info.height);
    out.data.resize(out.width * out.height * 3);

    for (int y = 0; y < out.height; ++y) {
        const uint8_t* row = pixels + y * info.stride;
        for (int x = 0; x < out.width; ++x) {
            int idx = (y * out.width + x) * 3;
            const uint8_t* px = row + x * 4;
            out.data[idx + 0] = px[0] / 255.0f; // R
            out.data[idx + 1] = px[1] / 255.0f; // G
            out.data[idx + 2] = px[2] / 255.0f; // B
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
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 || info.width != static_cast<uint32_t>(image.width) ||
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
        uint8_t* row = pixels + y * info.stride;
        for (int x = 0; x < image.width; ++x) {
            int idx = (y * image.width + x) * 3;
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

    float scale = static_cast<float>(targetWidth) / static_cast<float>(src.width);
    int targetHeight = static_cast<int>(std::round(src.height * scale));
    Image out;
    out.width = targetWidth;
    out.height = targetHeight;
    out.data.resize(out.width * out.height * 3);

    for (int y = 0; y < targetHeight; ++y) {
        float gy = (y + 0.5f) / scale - 0.5f;
        int y0 = static_cast<int>(std::floor(gy));
        int y1 = std::min(y0 + 1, src.height - 1);
        float dy = gy - y0;
        y0 = std::max(y0, 0);
        for (int x = 0; x < targetWidth; ++x) {
            float gx = (x + 0.5f) / scale - 0.5f;
            int x0 = static_cast<int>(std::floor(gx));
            int x1 = std::min(x0 + 1, src.width - 1);
            float dx = gx - x0;
            x0 = std::max(x0, 0);

            for (int c = 0; c < 3; ++c) {
                float v00 = src.data[(y0 * src.width + x0) * 3 + c];
                float v01 = src.data[(y0 * src.width + x1) * 3 + c];
                float v10 = src.data[(y1 * src.width + x0) * 3 + c];
                float v11 = src.data[(y1 * src.width + x1) * 3 + c];
                float v0 = v00 + (v01 - v00) * dx;
                float v1 = v10 + (v11 - v10) * dx;
                out.data[(y * targetWidth + x) * 3 + c] = v0 + (v1 - v0) * dy;
            }
        }
    }
    return out;
}

void computeCovariance(const Image& img, double mean[3], double cov[3][3]) {
    const size_t count = img.data.size() / 3;
    mean[0] = mean[1] = mean[2] = 0.0;
    for (size_t i = 0; i < count; ++i) {
        mean[0] += img.data[i * 3 + 0];
        mean[1] += img.data[i * 3 + 1];
        mean[2] += img.data[i * 3 + 2];
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
        double v0 = img.data[i * 3 + 0] - mean[0];
        double v1 = img.data[i * 3 + 1] - mean[1];
        double v2 = img.data[i * 3 + 2] - mean[2];
        cov[0][0] += v0 * v0;
        cov[0][1] += v0 * v1;
        cov[0][2] += v0 * v2;
        cov[1][0] += v1 * v0;
        cov[1][1] += v1 * v1;
        cov[1][2] += v1 * v2;
        cov[2][0] += v2 * v0;
        cov[2][1] += v2 * v1;
        cov[2][2] += v2 * v2;
    }
    double denom = static_cast<double>(count > 1 ? (count - 1) : 1);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            cov[r][c] /= denom;
        }
    }
}

void jacobiEigenDecomposition(double a[3][3], double eigenValues[3], double eigenVectors[3][3]) {
    // Initialize eigenvectors to identity
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            eigenVectors[i][j] = (i == j) ? 1.0 : 0.0;
        }
        eigenValues[i] = a[i][i];
    }

    const int maxIter = 24;
    for (int iter = 0; iter < maxIter; ++iter) {
        int p = 0, q = 1;
        double maxOff = std::fabs(a[0][1]);
        if (std::fabs(a[0][2]) > maxOff) {
            p = 0; q = 2; maxOff = std::fabs(a[0][2]);
        }
        if (std::fabs(a[1][2]) > maxOff) {
            p = 1; q = 2; maxOff = std::fabs(a[1][2]);
        }
        if (maxOff < 1e-10) break;

        double app = a[p][p];
        double aqq = a[q][q];
        double apq = a[p][q];
        double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        double c = std::cos(phi);
        double s = std::sin(phi);

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = a[q][p] = 0.0;

        for (int k = 0; k < 3; ++k) {
            if (k != p && k != q) {
                double akp = a[k][p];
                double akq = a[k][q];
                a[k][p] = a[p][k] = c * akp - s * akq;
                a[k][q] = a[q][k] = s * akp + c * akq;
            }
        }

        for (int k = 0; k < 3; ++k) {
            double vip = eigenVectors[k][p];
            double viq = eigenVectors[k][q];
            eigenVectors[k][p] = c * vip - s * viq;
            eigenVectors[k][q] = s * vip + c * viq;
        }
    }

    eigenValues[0] = a[0][0];
    eigenValues[1] = a[1][1];
    eigenValues[2] = a[2][2];
}

void applyPCADecorrelation(Image& img) {
    double mean[3];
    double cov[3][3];
    computeCovariance(img, mean, cov);

    double eigenValues[3];
    double eigenVectors[3][3];
    jacobiEigenDecomposition(cov, eigenValues, eigenVectors);

    const double eps = 1e-5;
    const size_t count = img.data.size() / 3;
    for (size_t i = 0; i < count; ++i) {
        double centered[3] = {
                img.data[i * 3 + 0] - mean[0],
                img.data[i * 3 + 1] - mean[1],
                img.data[i * 3 + 2] - mean[2]
        };
        double projected[3]{};
        for (int k = 0; k < 3; ++k) {
            projected[k] = eigenVectors[0][k] * centered[0] + eigenVectors[1][k] * centered[1] +
                            eigenVectors[2][k] * centered[2];
            projected[k] /= std::sqrt(std::fabs(eigenValues[k]) + eps);
        }
        double reconstructed[3]{};
        for (int j = 0; j < 3; ++j) {
            reconstructed[j] = eigenVectors[j][0] * projected[0] + eigenVectors[j][1] * projected[1] +
                               eigenVectors[j][2] * projected[2] + mean[j];
            img.data[i * 3 + j] = static_cast<float>(clamp01(static_cast<float>(reconstructed[j])));
        }
    }
}

// Color conversion utilities
inline float pivotRgb(float c) {
    return (c <= 0.04045f) ? (c / 12.92f) : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

inline float pivotXyz(float t) {
    return t > 0.008856f ? std::cbrt(t) : (7.787f * t + 16.0f / 116.0f);
}

void rgbToLab(float r, float g, float b, float& L, float& a, float& labB) {
    float rl = pivotRgb(r);
    float gl = pivotRgb(g);
    float bl = pivotRgb(b);

    float X = rl * 0.4124f + gl * 0.3576f + bl * 0.1805f;
    float Y = rl * 0.2126f + gl * 0.7152f + bl * 0.0722f;
    float Z = rl * 0.0193f + gl * 0.1192f + bl * 0.9505f;

    X /= 0.95047f;
    Y /= 1.0f;
    Z /= 1.08883f;

    float fx = pivotXyz(X);
    float fy = pivotXyz(Y);
    float fz = pivotXyz(Z);

    L = 116.0f * fy - 16.0f;
    a = 500.0f * (fx - fy);
    labB = 200.0f * (fy - fz);
}

inline float invPivotXyz(float t) {
    float t3 = t * t * t;
    return t3 > 0.008856f ? t3 : (t - 16.0f / 116.0f) / 7.787f;
}

void labToRgb(float L, float a, float labB, float& r, float& g, float& b) {
    float fy = (L + 16.0f) / 116.0f;
    float fx = a / 500.0f + fy;
    float fz = fy - labB / 200.0f;

    float X = invPivotXyz(fx) * 0.95047f;
    float Y = invPivotXyz(fy) * 1.0f;
    float Z = invPivotXyz(fz) * 1.08883f;

    float rl = X * 3.2406f + Y * -1.5372f + Z * -0.4986f;
    float gl = X * -0.9689f + Y * 1.8758f + Z * 0.0415f;
    float bl = X * 0.0557f + Y * -0.2040f + Z * 1.0570f;

    auto gamma = [](float c) {
        return c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
    };

    r = clamp01(gamma(rl));
    g = clamp01(gamma(gl));
    b = clamp01(gamma(bl));
}

std::vector<uint8_t> buildCLAHELut(const std::vector<float>& luma, int startX, int startY, int tileW, int tileH,
                                   int width, int height, float clipLimit) {
    const int bins = 256;
    std::vector<int> hist(bins, 0);
    int endX = std::min(startX + tileW, width);
    int endY = std::min(startY + tileH, height);
    int area = (endX - startX) * (endY - startY);

    for (int y = startY; y < endY; ++y) {
        for (int x = startX; x < endX; ++x) {
            int idx = y * width + x;
            int bin = std::clamp(static_cast<int>(luma[idx] * 255.0f + 0.5f), 0, 255);
            hist[bin]++;
        }
    }

    if (clipLimit > 0.0f) {
        int limit = static_cast<int>(clipLimit * static_cast<float>(area) / bins) + 1;
        int excess = 0;
        for (int i = 0; i < bins; ++i) {
            if (hist[i] > limit) {
                excess += hist[i] - limit;
                hist[i] = limit;
            }
        }
        int increment = excess / bins;
        int remainder = excess % bins;
        for (int i = 0; i < bins; ++i) {
            hist[i] += increment;
            if (i < remainder) hist[i] += 1;
        }
    }

    std::vector<uint8_t> lut(bins, 0);
    int cumulative = 0;
    for (int i = 0; i < bins; ++i) {
        cumulative += hist[i];
        lut[i] = static_cast<uint8_t>(std::clamp((cumulative * 255) / std::max(area, 1), 0, 255));
    }
    return lut;
}

void applyCLAHE(std::vector<float>& luma, int width, int height, int tilesX = 8, int tilesY = 8, float clipLimit = 2.0f) {
    int tileW = (width + tilesX - 1) / tilesX;
    int tileH = (height + tilesY - 1) / tilesY;

    std::vector<std::vector<uint8_t>> luts(tilesX * tilesY);
    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            luts[ty * tilesX + tx] = buildCLAHELut(luma, tx * tileW, ty * tileH, tileW, tileH, width, height, clipLimit);
        }
    }

    for (int y = 0; y < height; ++y) {
        float gy = (static_cast<float>(y) + 0.5f) / tileH - 0.5f;
        int ty = static_cast<int>(std::floor(gy));
        int ty1 = std::min(ty + 1, tilesY - 1);
        ty = std::clamp(ty, 0, tilesY - 1);
        float dy = gy - ty;

        for (int x = 0; x < width; ++x) {
            float gx = (static_cast<float>(x) + 0.5f) / tileW - 0.5f;
            int tx = static_cast<int>(std::floor(gx));
            int tx1 = std::min(tx + 1, tilesX - 1);
            tx = std::clamp(tx, 0, tilesX - 1);
            float dx = gx - tx;

            int idx = y * width + x;
            int bin = std::clamp(static_cast<int>(luma[idx] * 255.0f + 0.5f), 0, 255);

            auto lut00 = luts[ty * tilesX + tx][bin];
            auto lut01 = luts[ty * tilesX + tx1][bin];
            auto lut10 = luts[ty1 * tilesX + tx][bin];
            auto lut11 = luts[ty1 * tilesX + tx1][bin];

            float interpTop = lut00 + (lut01 - lut00) * dx;
            float interpBottom = lut10 + (lut11 - lut10) * dx;
            float finalVal = interpTop + (interpBottom - interpTop) * dy;

            luma[idx] = clamp01(finalVal / 255.0f);
        }
    }
}

Image applyLabClahe(const Image& img) {
    Image out;
    out.width = img.width;
    out.height = img.height;
    out.data.resize(img.data.size());

    std::vector<float> L(img.width * img.height);
    std::vector<float> a(img.width * img.height);
    std::vector<float> b(img.width * img.height);

    for (int i = 0; i < img.width * img.height; ++i) {
        float r = img.data[i * 3 + 0];
        float g = img.data[i * 3 + 1];
        float bl = img.data[i * 3 + 2];
        float lVal, aVal, bVal;
        rgbToLab(r, g, bl, lVal, aVal, bVal);
        L[i] = lVal / 100.0f; // normalize 0..1
        a[i] = aVal;
        b[i] = bVal;
    }

    applyCLAHE(L, img.width, img.height);

    for (int i = 0; i < img.width * img.height; ++i) {
        float lScaled = std::clamp(L[i] * 100.0f, 0.0f, 100.0f);
        float r, g, bl;
        labToRgb(lScaled, a[i], b[i], r, g, bl);
        out.data[i * 3 + 0] = r;
        out.data[i * 3 + 1] = g;
        out.data[i * 3 + 2] = bl;
    }

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
        jint jpegQuality) {
    const char* inPathChars = env->GetStringUTFChars(jInputPath, nullptr);
    const char* outPathChars = env->GetStringUTFChars(jOutputPath, nullptr);
    std::string inputPath(inPathChars ? inPathChars : "");
    std::string outputPath(outPathChars ? outPathChars : "");
    env->ReleaseStringUTFChars(jInputPath, inPathChars);
    env->ReleaseStringUTFChars(jOutputPath, outPathChars);

    jclass optionsCls = env->FindClass("android/graphics/BitmapFactory$Options");
    jmethodID optionsCtor = env->GetMethodID(optionsCls, "<init>", "()V");
    jobject optionsObj = env->NewObject(optionsCls, optionsCtor);
    jfieldID inPreferredConfigField = env->GetFieldID(optionsCls, "inPreferredConfig",
                                                     "Landroid/graphics/Bitmap$Config;");

    jclass bitmapConfigCls = env->FindClass("android/graphics/Bitmap$Config");
    jmethodID valueOfConfig = env->GetStaticMethodID(bitmapConfigCls, "valueOf",
                                                    "(Ljava/lang/String;)Landroid/graphics/Bitmap$Config;");
    jstring argbStr = env->NewStringUTF("ARGB_8888");
    jobject argbConfig = env->CallStaticObjectMethod(bitmapConfigCls, valueOfConfig, argbStr);
    env->SetObjectField(optionsObj, inPreferredConfigField, argbConfig);
    env->DeleteLocalRef(argbStr);

    jclass bfCls = env->FindClass("android/graphics/BitmapFactory");
    jmethodID decodeMethod = env->GetStaticMethodID(bfCls, "decodeFile",
                                                    "(Ljava/lang/String;Landroid/graphics/BitmapFactory$Options;)Landroid/graphics/Bitmap;");
    jstring inputJ = env->NewStringUTF(inputPath.c_str());
    jobject bitmap = env->CallStaticObjectMethod(bfCls, decodeMethod, inputJ, optionsObj);
    env->DeleteLocalRef(inputJ);

    if (bitmap == nullptr) {
        ALOGE("Failed to decode bitmap at %s", inputPath.c_str());
        return -1;
    }

    Image image{};
    if (!bitmapToImage(env, bitmap, image)) {
        env->DeleteLocalRef(bitmap);
        return -2;
    }

    int desiredWidth = targetWidth > 0 ? targetWidth : 1980;
    Image resized = resizeBilinear(image, desiredWidth);
    applyPCADecorrelation(resized);
    Image finalImg = applyLabClahe(resized);

    jclass bitmapCls = env->FindClass("android/graphics/Bitmap");
    jmethodID createBitmapMethod = env->GetStaticMethodID(bitmapCls, "createBitmap",
                                                         "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    jobject outBitmap = env->CallStaticObjectMethod(bitmapCls, createBitmapMethod,
                                                   resized.width, resized.height, argbConfig);

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
    jstring outPathJ = env->NewStringUTF(outputPath.c_str());
    jobject fos = env->NewObject(fosCls, fosCtor, outPathJ);
    env->DeleteLocalRef(outPathJ);

    jclass compressFormatCls = env->FindClass("android/graphics/Bitmap$CompressFormat");
    jfieldID jpegField = env->GetStaticFieldID(compressFormatCls, "JPEG",
                                              "Landroid/graphics/Bitmap$CompressFormat;");
    jobject jpegFormat = env->GetStaticObjectField(compressFormatCls, jpegField);

    jmethodID compressMethod = env->GetMethodID(bitmapCls, "compress",
                                               "(Landroid/graphics/Bitmap$CompressFormat;ILjava/io/OutputStream;)Z");
    jboolean compressed = env->CallBooleanMethod(outBitmap, compressMethod, jpegFormat,
                                                 jpegQuality > 0 ? jpegQuality : 50, fos);

    jmethodID closeMethod = env->GetMethodID(fosCls, "close", "()V");
    env->CallVoidMethod(fos, closeMethod);

    env->DeleteLocalRef(bitmap);
    env->DeleteLocalRef(outBitmap);
    env->DeleteLocalRef(fos);

    return compressed ? 0 : -5;
}
