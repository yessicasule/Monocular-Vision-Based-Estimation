// MonoArmManager.cs
// ===================
// Main scene manager. Wires UdpAngleReceiver → AvatarMuscleController(s)
// and displays live status in the Debug Console and on a world-space HUD.
//
// Bilateral support: assign both avatarController (side = Right) and
// leftArmController (side = Left) to drive both arms from a 'B,' packet.
// leftArmController is optional — leaving it unassigned (and the scene
// without a second AvatarMuscleController) reproduces the original
// single-arm ('S,' packet) behavior exactly.
//
// Setup
// -----
// 1. Create an empty GameObject named "MonoArmManager".
// 2. Attach this component.
// 3. Assign the UdpAngleReceiver and AvatarMuscleController reference(s)
//    (or run MonoArm → Build Scene from the menu to do it automatically).

using UnityEngine;

namespace MonoArm
{
    public class MonoArmManager : MonoBehaviour
    {
        [Header("References")]
        public UdpAngleReceiver    receiver;
        public AvatarMuscleController avatarController;      // side = Right (or the sole arm in single-arm scenes)
        public AvatarMuscleController leftArmController;     // side = Left; optional, bilateral only

        [Header("Status")]
        [Tooltip("Log angle values to the console every N seconds (0 = disabled).")]
        public float logIntervalSeconds = 0f;

        float _logTimer;

        void Start()
        {
            if (receiver == null)
                receiver = FindFirstObjectByType<UdpAngleReceiver>();

            if (avatarController == null || leftArmController == null)
            {
                foreach (var c in FindObjectsByType<AvatarMuscleController>(FindObjectsSortMode.None))
                {
                    if (c.side == ArmSide.Right && avatarController == null) avatarController = c;
                    else if (c.side == ArmSide.Left && leftArmController == null) leftArmController = c;
                }
            }

            if (receiver == null)
                Debug.LogError("[MonoArmManager] UdpAngleReceiver not found in scene.");

            if (avatarController == null)
                Debug.LogError("[MonoArmManager] AvatarMuscleController (Right/primary) not found in scene.");

            if (leftArmController == null)
                Debug.Log("[MonoArmManager] No left-arm AvatarMuscleController assigned — left-arm data " +
                          "from bilateral packets will be received but not applied to any avatar.");
        }

        // LateUpdate, not Update: Unity evaluates the Animator between the two,
        // and an Animator playing a clip overwrites bone writes made before it
        // runs. Verified on this rig — applying angles before the animation
        // evaluates loses them entirely (45°/20°/90° came back as the rig's
        // neutral 29.6°/41.4°/80°), while applying after preserves them exactly.
        // Harmless today (the scene has no AnimatorController) but required the
        // moment any animation clip is added.
        void LateUpdate()
        {
            if (receiver == null || avatarController == null) return;
            if (!receiver.HasData) return;

            // Forward right-arm (or the sole arm, for 'S,' packets) angles.
            avatarController.ApplyAngles(receiver.LatestAngles);

            // Forward left-arm angles when the latest packet was bilateral.
            if (receiver.IsBilateral && leftArmController != null)
                leftArmController.ApplyAngles(receiver.LatestBilateralAngles.left);

            LogStatusIfDue();
        }

        void LogStatusIfDue()
        {
            if (logIntervalSeconds <= 0f) return;

            _logTimer += Time.unscaledDeltaTime;
            if (_logTimer < logIntervalSeconds) return;
            _logTimer = 0f;

            ArmAngles r = receiver.LatestAngles;
            string msg = $"[MonoArmManager] pkts={receiver.PacketCount} " +
                         $"R[{r}]";
            if (receiver.IsBilateral && leftArmController != null)
                msg += $" L[{receiver.LatestBilateralAngles.left}]";
            Debug.Log(msg);
        }
    }
}
