// AvatarMuscleController.cs
// ==========================
// Controls one of the humanoid avatar's arms (Right or Left, selectable via
// the `side` field) using Unity's HumanPoseHandler and the muscle-space API.
// This approach is avatar-agnostic: it works with any Humanoid-rigged model
// without hardcoding bone names or local axes. For bilateral tracking, add
// two instances of this component to the same Animator, one per side.
//
// Why not muscle space
// --------------------
// An earlier version drove Unity's normalised [−1, +1] muscle values, one
// muscle per incoming angle. That cannot work, and the rig was measured to
// confirm it: sweeping "Right Arm Down-Up" from −1 to +1 moves BOTH angles
// (flexion −2°→168°, abduction −11°→63°→29°), and sweeping "Arm Front-Back"
// likewise moves both. Abduction is non-monotonic in each muscle, so there
// is no invertible one-to-one map from an angle to a muscle value — a target
// abduction of 40° has two muscle solutions on the same axis.
//
// The two shoulder muscles are simply not the same basis as the solver's
// flexion/abduction decomposition. Rather than fit a coupled 2-D inverse,
// this component reconstructs the upper-arm and forearm direction vectors
// analytically (the exact inverse of angle_solver.py) and aligns the bones
// to them. That is correct by construction and needs no per-rig calibration.
//
// Elbow and rotation WERE clean single DOFs when measured (elbow: exactly
// linear, 80° − 80·muscle, fully decoupled; rotation: exactly linear at
// 90°/unit), but they are reconstructed the same way here for consistency.
//
//
// Reconstruction
// --------------
// For the right arm, angle_solver.py decomposes the unit upper-arm vector
// v = (x_lateral, y_superior, z_anterior) in the torso frame as
//
//   abduction = asin(-vx)          flexion = atan2(vz, -vy)
//
// so a straight-down arm is (0, -1, 0). Inverting that:
//
//   vx = -sin(A)   vy = -cos(A)cos(F)   vz = cos(A)sin(F)
//
// The torso frame must be built with anatomical axes (x = subject's left,
// y = up, z = anterior). Unity is left-handed, so z = Cross(y, x) here,
// not the Cross(x, y) used in the right-handed Python code — see
// BuildTorsoFrame.
//
// The forearm is then placed at the elbow's included angle from the upper
// arm, swung about it by the rotation angle in the same (e1, e2) basis the
// solver builds, so decomposing the result returns the same four angles.
// The left arm is reflected across the sagittal plane (negate x), matching
// the mirror angle_solver.compute_bilateral_angles() applies.
//
// Bones are aligned with Quaternion.FromToRotation, upper arm first, then
// the forearm (which has moved with its parent).
//
// Two consequences, both measured on this rig rather than assumed:
//
//   * The Avatar's muscle limits no longer clamp anything. Muscle space tops
//     out at 160° of elbow flexion; asking this component for 175° yields
//     exactly 175°. Anatomically impossible input will be rendered faithfully,
//     so clamp upstream if that matters.
//
//   * Writes must happen AFTER the Animator evaluates, which is why
//     MonoArmManager drives this from LateUpdate. With a clip playing,
//     applying angles first loses them completely (45/20/90 came back as the
//     rig's neutral 29.6/41.4/80); applying afterwards preserves them exactly.
//
// Note that abduction is only uniquely defined on [-90, +90] — it comes from
// asin(), so angle_solver.py can never emit anything outside that band, and a
// value beyond it aliases back (170° reads back as 10°).

using UnityEngine;

namespace MonoArm
{
    public enum ArmSide { Right, Left }

    [RequireComponent(typeof(Animator))]
    public class AvatarMuscleController : MonoBehaviour
    {
        // ── Inspector ───────────────────────────────────────────────────────

        [Header("Side")]
        [Tooltip("Which arm this controller instance drives. Add one instance per side for bilateral tracking.")]
        public ArmSide side = ArmSide.Right;

        [Header("Smoothing")]
        [Tooltip("SmoothDamp time for each muscle value. Lower = faster but jitterier.")]
        [Range(0.02f, 0.5f)]
        public float smoothTime = 0.08f;

        [Header("Calibration Offsets (degrees)")]
        [Tooltip("Subtracted from incoming flexion before it is applied. Use to zero the neutral pose.")]
        public float flexionOffset   = 0f;
        public float abductionOffset = 0f;
        [Tooltip("Shoulder rotation is a forearm-proxy angle with no absolute zero — " +
                 "calibrate this to the subject's neutral-pose reading.")]
        public float rotationOffset  = 0f;
        public float elbowOffset     = 0f;

        // ── Private ─────────────────────────────────────────────────────────

        Animator  _anim;
        Transform _upper, _lower, _hand;
        Transform _lShoulder, _rShoulder, _lHip, _rHip;
        bool      _ready;

        // Smoothed angle state (degrees) and SmoothDamp velocities
        float _sFlex, _sAbd, _sRot, _sElbow;
        float _vFlex, _vAbd, _vRot, _vElbow;

        // Last applied angles (for the inspector readout)
        public float CurrentFlexionDeg   { get; private set; }
        public float CurrentAbductionDeg { get; private set; }
        public float CurrentRotationDeg  { get; private set; }
        public float CurrentElbowDeg     { get; private set; }

        // ── Unity Lifecycle ─────────────────────────────────────────────────

        void Awake()
        {
            _anim = GetComponent<Animator>();
            if (_anim == null || !_anim.isHuman)
            {
                Debug.LogError("[AvatarMuscleController] Animator not found or not Humanoid. " +
                               "Set Animation Type to Humanoid in the FBX import settings.");
                enabled = false;
                return;
            }

            bool right = side == ArmSide.Right;
            _upper = _anim.GetBoneTransform(right ? HumanBodyBones.RightUpperArm : HumanBodyBones.LeftUpperArm);
            _lower = _anim.GetBoneTransform(right ? HumanBodyBones.RightLowerArm : HumanBodyBones.LeftLowerArm);
            _hand  = _anim.GetBoneTransform(right ? HumanBodyBones.RightHand     : HumanBodyBones.LeftHand);

            _lShoulder = _anim.GetBoneTransform(HumanBodyBones.LeftUpperArm);
            _rShoulder = _anim.GetBoneTransform(HumanBodyBones.RightUpperArm);
            _lHip      = _anim.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
            _rHip      = _anim.GetBoneTransform(HumanBodyBones.RightUpperLeg);

            _ready = _upper && _lower && _hand && _lShoulder && _rShoulder && _lHip && _rHip;
            if (!_ready)
            {
                Debug.LogError($"[AvatarMuscleController:{side}] Required humanoid bones missing " +
                               "(upper arm / lower arm / hand / shoulders / hips).");
                enabled = false;
                return;
            }

            Debug.Log($"[AvatarMuscleController:{side}] Bones resolved — driving " +
                      $"{_upper.name} / {_lower.name} directly.");
        }

        // ── Public API ──────────────────────────────────────────────────────

        /// <summary>
        /// Apply a set of anatomical arm angles to this controller's arm
        /// (<see cref="side"/>). Call this once per frame from MonoArmManager,
        /// once per side/controller instance.
        /// </summary>
        /// <param name="angles">Incoming joint angles (degrees) from the UDP receiver.</param>
        public void ApplyAngles(ArmAngles angles)
        {
            if (!_ready) return;

            float flex  = angles.shoulderFlexion   - flexionOffset;
            float abd   = angles.shoulderAbduction - abductionOffset;
            float rot   = angles.shoulderRotation  - rotationOffset;
            float elbow = angles.elbowFlexion      - elbowOffset;

            // Smooth in angle space. SmoothDampAngle wraps correctly at ±180°,
            // which matters for flexion (arm overhead sits near the wrap).
            float dt = Time.deltaTime;
            if (smoothTime > 0f && dt > 0f)
            {
                _sFlex  = Mathf.SmoothDampAngle(_sFlex,  flex,  ref _vFlex,  smoothTime, Mathf.Infinity, dt);
                _sAbd   = Mathf.SmoothDampAngle(_sAbd,   abd,   ref _vAbd,   smoothTime, Mathf.Infinity, dt);
                _sRot   = Mathf.SmoothDampAngle(_sRot,   rot,   ref _vRot,   smoothTime, Mathf.Infinity, dt);
                _sElbow = Mathf.SmoothDampAngle(_sElbow, elbow, ref _vElbow, smoothTime, Mathf.Infinity, dt);
            }
            else
            {
                _sFlex = flex; _sAbd = abd; _sRot = rot; _sElbow = elbow;
            }

            BuildTorsoFrame(out Vector3 tx, out Vector3 ty, out Vector3 tz);

            // Reconstruct the upper-arm direction, then align the bone to it.
            Vector3 uT = UpperArmDirection(_sFlex, _sAbd);
            Vector3 fT = ForearmDirection(uT, _sElbow, _sRot);
            if (side == ArmSide.Left)
            {
                // angle_solver mirrors the left arm across the sagittal plane
                // before decomposing, so undo that reflection here.
                uT.x = -uT.x;
                fT.x = -fT.x;
            }

            Vector3 uW = tx * uT.x + ty * uT.y + tz * uT.z;
            Vector3 curU = (_lower.position - _upper.position).normalized;
            if (curU.sqrMagnitude > 1e-12f && uW.sqrMagnitude > 1e-12f)
                _upper.rotation = Quaternion.FromToRotation(curU, uW) * _upper.rotation;

            // The forearm bone has moved with its parent — re-read it before aligning.
            Vector3 fW = tx * fT.x + ty * fT.y + tz * fT.z;
            Vector3 curF = (_hand.position - _lower.position).normalized;
            if (curF.sqrMagnitude > 1e-12f && fW.sqrMagnitude > 1e-12f)
                _lower.rotation = Quaternion.FromToRotation(curF, fW) * _lower.rotation;

            CurrentFlexionDeg   = _sFlex;
            CurrentAbductionDeg = _sAbd;
            CurrentRotationDeg  = _sRot;
            CurrentElbowDeg     = _sElbow;
        }

        // ── Private helpers ──────────────────────────────────────────────────

        /// <summary>
        /// Build the avatar's torso frame exactly as coordinate_frame.py does:
        /// y = hips→shoulders, x = toward the subject's LEFT, z = anterior.
        /// Body-relative, so it is invariant to how the avatar is oriented.
        /// </summary>
        void BuildTorsoFrame(out Vector3 x, out Vector3 y, out Vector3 z)
        {
            Vector3 hipMid = 0.5f * (_lHip.position + _rHip.position);
            Vector3 shMid  = 0.5f * (_lShoulder.position + _rShoulder.position);

            y = (shMid - hipMid).normalized;
            Vector3 xCand = (_lShoulder.position - _rShoulder.position).normalized;
            x = (xCand - Vector3.Dot(xCand, y) * y).normalized;
            // Unity's world is left-handed, so the cross-product order that
            // gives the anterior axis in coordinate_frame.py (right-handed,
            // x × y) points BEHIND the avatar here. y × x is anterior in Unity.
            // With (x, y, z) = (subject's left, up, anterior) the torso-frame
            // coordinates are numerically identical to angle_solver.py's.
            z = Vector3.Cross(y, x).normalized;
        }

        /// <summary>
        /// Inverse of angle_solver._compute_shoulder_angles for the right arm.
        /// Given flexion and abduction, return the unit upper-arm vector in the
        /// torso frame. Forward map: abduction = asin(−vx),
        /// flexion = atan2(vz, −vy), so a straight-down arm is (0, −1, 0)
        /// and positive flexion swings it anterior (+z).
        /// </summary>
        static Vector3 UpperArmDirection(float flexionDeg, float abductionDeg)
        {
            float f = flexionDeg   * Mathf.Deg2Rad;
            float a = abductionDeg * Mathf.Deg2Rad;
            float c = Mathf.Cos(a);
            return new Vector3(-Mathf.Sin(a), -c * Mathf.Cos(f), c * Mathf.Sin(f)).normalized;
        }

        /// <summary>
        /// Inverse of angle_solver._compute_shoulder_rotation / elbow flexion.
        /// The forearm sits at <paramref name="elbowDeg"/> from the upper arm,
        /// swung around it by <paramref name="rotationDeg"/> in the same (e1, e2)
        /// basis the solver uses, so the round trip reproduces both angles.
        /// </summary>
        static Vector3 ForearmDirection(Vector3 u, float elbowDeg, float rotationDeg)
        {
            u = u.normalized;
            Vector3 reference = new Vector3(1f, 0f, 0f);
            if (Mathf.Abs(Vector3.Dot(u, reference)) > 0.9f) reference = new Vector3(0f, 0f, 1f);

            Vector3 e1 = Vector3.Cross(u, reference);
            if (e1.sqrMagnitude < 1e-18f) return u;
            e1 = e1.normalized;
            Vector3 e2 = Vector3.Cross(u, e1);

            float e = elbowDeg    * Mathf.Deg2Rad;
            float r = rotationDeg * Mathf.Deg2Rad;
            Vector3 perp = Mathf.Sin(r) * e1 + Mathf.Cos(r) * e2;
            return (Mathf.Cos(e) * u + Mathf.Sin(e) * perp).normalized;
        }
    }
}
