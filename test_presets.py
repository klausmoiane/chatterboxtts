from voice_templates.preset_manager import get_preset_names, list_presets

print("Testing preset loading...")
print("=" * 50)

presets = list_presets()
print(f"Total presets found: {len(presets)}")
print()

for preset in presets:
    print(f"Preset: {preset.get('name')}")
    print(f"  File stem: {preset.get('_file_stem')}")
    print(f"  Audio ref: {preset.get('audio_ref')}")
    print()

names = get_preset_names()
print(f"Preset names: {names}")
print()

if names:
    print("✅ Presets loaded successfully!")
else:
    print("❌ No presets found - dropdown will be empty!")
