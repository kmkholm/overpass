"""OVERPASS: forensic verification of audio recording location from propeller-aircraft
Doppler signatures and public ADS-B tracks.

Built on top of the open dopplertrack proof of concept (MIT). Four additions:
  synth / detect   automatic flyover detection (ridge segmenter trained on spectrograms
                   rendered from real ADS-B tracks), no hand-chosen windows
  contrast         contrastive locked scoring (curve contrast z-scored against its
                   neighbourhood) in place of raw brightness
  nsearch          joint search over N flyovers with the silence constraint
  conformal        calibrated location sets (coverage guarantee) instead of one pin
  scrub            the countermeasure: track-and-fill removal of the Doppler comb
"""
__version__ = "0.1.0"
