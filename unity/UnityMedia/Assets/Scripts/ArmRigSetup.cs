#if UNITY_EDITOR
// ArmRigSetup.cs
// ===============
// Custom Inspector for AvatarMuscleController.
// Adds a diagnostic button that reports resolved muscle indices
// and shows live muscle values in the Editor during Play mode.

using UnityEditor;
using UnityEngine;

namespace MonoArm
{
    [CustomEditor(typeof(AvatarMuscleController))]
    public class ArmRigSetup : Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            var ctrl = (AvatarMuscleController)target;
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("MonoArm Diagnostics", EditorStyles.boldLabel);

            if (Application.isPlaying)
            {
                EditorGUILayout.LabelField("Live Applied Angles (deg)", EditorStyles.miniLabel);
                using (new EditorGUI.DisabledScope(true))
                {
                    EditorGUILayout.FloatField("Flexion",   ctrl.CurrentFlexionDeg);
                    EditorGUILayout.FloatField("Abduction", ctrl.CurrentAbductionDeg);
                    EditorGUILayout.FloatField("Rotation",  ctrl.CurrentRotationDeg);
                    EditorGUILayout.FloatField("Elbow",     ctrl.CurrentElbowDeg);
                }
                Repaint();   // refresh every frame during play
            }
            else
            {
                EditorGUILayout.HelpBox(
                    "Muscle values shown here during Play mode.\n\n" +
                    "Setup instructions:\n" +
                    "1. Ensure your avatar FBX uses Animation Type = Humanoid.\n" +
                    "2. Run  MonoArm → Build Scene  to wire all components.\n" +
                    "3. Start the Python pipeline, then press Play.",
                    MessageType.Info);
            }
        }
    }
}
#endif
