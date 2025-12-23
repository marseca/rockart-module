# Proyecto: RockEnhancer (Android NDK)

Objetivo: implementar módulo Android (React Native NativeModule) que procese imágenes:
- input: URI o path a imagen
- output: path a JPG preview 1980px ancho, calidad 50
- procesamiento: PCA decorrelation + CLAHE (aplicar CLAHE solo en luminancia Lab)

Restricciones:
- No usar base64 entre RN y nativo: solo rutas de fichero.
- No bloquear el hilo UI: ejecutar en background.
- Mantener cambios dentro de /android (y si hace falta, añadir un wrapper RN mínimo).

Entregables:
- NativeModule Kotlin: RockEnhancerModule con método processPreview(...)
- NDK C++: función nativeProcess(...) (C++17) + CMakeLists + integración JNI
- Instrucciones para compilar: ./gradlew :app:assembleDebug
- Tests manuales: al menos 3 fotos, sin crash y con output generado en cache.
