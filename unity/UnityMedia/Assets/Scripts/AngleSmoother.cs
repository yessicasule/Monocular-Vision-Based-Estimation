// AngleSmoother.cs
// =================
// Simple per-channel exponential smoother using Mathf.SmoothDampAngle.
// Used inside AvatarMuscleController as a fallback for any values
// that still need separate smoothing outside the muscle pipeline.
//
// Note: The primary smoothing in MonoArm is done via SmoothDamp
// directly on muscle values inside AvatarMuscleController. This class
// is retained for optional external use (e.g. debug overlay angles).

using UnityEngine;

namespace MonoArm
{
    public class AngleSmoother
    {
        float _current;
        float _velocity;
        readonly float _smoothTime;
        bool  _init;

        /// <param name="smoothTime">
        /// Approximate time (seconds) to reach the target value.
        /// Smaller = more responsive; larger = smoother.
        /// </param>
        public AngleSmoother(float smoothTime = 0.08f)
        {
            _smoothTime = smoothTime;
        }

        /// <summary>Feed a new raw angle (degrees) and get the smoothed value.</summary>
        public float Update(float rawAngle)
        {
            if (!_init)
            {
                _current = rawAngle;
                _init    = true;
                return rawAngle;
            }
            _current = Mathf.SmoothDampAngle(_current, rawAngle, ref _velocity,
                                              _smoothTime, Mathf.Infinity, Time.deltaTime);
            return _current;
        }

        public void Reset()
        {
            _init     = false;
            _velocity = 0f;
        }

        public float Current => _current;
    }
}
