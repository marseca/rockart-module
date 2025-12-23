import { requireNativeModule } from "expo";

// import { RockenhancerModuleEvents } from "./Rockenhancer.types"

declare class RockenhancerModule {
	processPreview(
		inputUriOrPath: string,
		targetWidth: number,
		jpegQuality: number
	): Promise<string>;
}

// This call loads the native module object from the JSI.
export default requireNativeModule<RockenhancerModule>("Rockenhancer");
