package expo.modules.rockenhancer

import android.content.ContentResolver
import android.net.Uri
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class RockenhancerModule : Module() {

	private val executor = Executors.newSingleThreadExecutor()

	override fun definition() = ModuleDefinition {
	Name("Rockenhancer")

	AsyncFunction("processPreview") { inputUriOrPath: String, targetWidth: Int, jpegQuality: Int, mode: String, factors: Array<Float> ->
			val safeTargetWidth = if (targetWidth > 0) targetWidth else 1980
			val safeQuality = if (jpegQuality in 1..100) jpegQuality else 50
			val safeMode = if (mode != "") mode else "yxx"
			val safeFactors = if (factors.isNotEmpty()) factors else arrayOf(1.0f, 1.0f, 1.0f)	// TODO SPlit to 3 lenght array

			val ctx = appContext.reactContext?.applicationContext
			?: throw IllegalStateException("React context not available")

			// Run work in background thread, but return the String result to JS
			val future = executor.submit(Callable {
				val inputFile = materializeInput(ctx.contentResolver, ctx.cacheDir, inputUriOrPath)
				val outputFile = File(ctx.cacheDir, "out_${System.currentTimeMillis()}.jpg")

				val result = nativeProcess(
					inputFile.absolutePath,
					outputFile.absolutePath,
					safeTargetWidth,
					safeQuality,
					safeMode,
					safeFactors
				)
				if (result != 0) throw IOException("Native processing failed (code=$result)")

				"file://${outputFile.absolutePath}"
			})

			// Wait for completion (OK because we're not on the JS thread here)
			future.get(10, TimeUnit.MINUTES)
		}
	}

	private fun materializeInput(
		resolver: ContentResolver,
		cacheDir: File,
		input: String
	): File {
		val uri = Uri.parse(input)

		return if (uri.scheme == "content") {
			val target = File(cacheDir, "input_${System.currentTimeMillis()}.jpg")
			resolver.openInputStream(uri)?.use { inputStream ->
				FileOutputStream(target).use { out -> inputStream.copyTo(out) }
			} ?: throw IOException("Unable to open input stream for $input")
			target
		} else {
			val path = if (input.startsWith("file://")) Uri.parse(input).path else input
			val file = File(path ?: input)
			if (!file.exists()) throw IOException("Input file does not exist: $input")
			file
		}
	}

	private external fun nativeProcess(
		inputPath: String,
		outputPath: String,
		targetWidth: Int,
		jpegQuality: Int,
		mode: String,
		factors: Array<Float>
	): Int

	companion object {
		init {
			System.loadLibrary("rockenhancer")
		}
	}
}
