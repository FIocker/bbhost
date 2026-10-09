# DLSS / 2x FG pacing evidence

Hunter's Dream, copied Bloodborne 1.09 save; RTX 5080 / driver 617.14.
DLSS Quality, object motion enabled, 60 FPS game cap and unchanged 116 FPS
driver cap. The requested internal output is about 7.0 megapixels;
the captured client surface is 1920x1080.

before.png: camera-sweep failure in the previous capped-jitter phase filter.
after.png: corrected bounded, unbiased phase in the packaged build.
Same location, slightly different camera angle. Both are unmodified client
captures. The counter is illustrative; measured display cadence comes from
PresentMon and is documented on render/dlss-sr-framegen-pipeline.

Renderer source: 3a49e997f059c5bbcbd2e401234c7b72d6d1e72e.
Local packaged source also includes the separate chromatic-aberration fix;
chromatic aberration remains enabled in both captures.

This branch contains rendered screenshots only, no dump, extracted game
assets, eboot, save or account credentials.
