#if UNITY_EDITOR
// SceneBuilder.cs
// ================
// Editor utility to automatically build the MonoArm scene.
//
// Menu: MonoArm / Build Scene
//   1. Strips the scene down to a single humanoid: destroys every extra
//      avatar, comparison Label_* object, and root Canvas left over from
//      multi-framework comparison scenes (e.g. the MediaPipe/MoveNet/
//      DeepFusionPose/GAN comparison view), and collapses duplicate
//      PoseManager/MonoArmManager objects.
//   2. Attaches AvatarMuscleController, UdpAngleReceiver, MonoArmManager,
//      and PoseDebugUI to a PoseManager GameObject driving the one
//      remaining avatar.
//   3. Positions the camera for a clear avatar view.
//   4. Saves the scene.
//
// Menu: MonoArm / Diagnose Scene
//   Reports the status of all required components to the Console.
//
// Menu: MonoArm / Undo Last Build
//   Undoes the last Build Scene operation.

using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace MonoArm
{
    public static class SceneBuilder
    {
        // ── Build ────────────────────────────────────────────────────────────

        [MenuItem("MonoArm/Build Scene")]
        static void Build()
        {
            Undo.IncrementCurrentGroup();
            Undo.SetCurrentGroupName("MonoArm Build Scene");
            int undoGroup = Undo.GetCurrentGroup();

            CollapseToSingleAvatar();

            // Remove old MonoArm manager objects (not the avatar) — including
            // any duplicates left behind by repeated builds.
            DestroyAllNamed("PoseManager");
            DestroyAllNamed("MonoArmManager");

            // Find or instantiate humanoid avatar
            GameObject avatar = FindHumanoidInScene();
            if (avatar == null)
            {
                // Try X Bot fallback
                string[] guids = AssetDatabase.FindAssets("X Bot t:GameObject");
                if (guids.Length == 0)
                {
                    Undo.CollapseUndoOperations(undoGroup);
                    EditorUtility.DisplayDialog("MonoArm — No Avatar Found",
                        "No Humanoid avatar is in the scene and no 'X Bot' asset was found.\n\n" +
                        "Please drag a Humanoid-rigged FBX into the scene, then run Build Scene again.\n\n" +
                        "You can get free avatars from Mixamo (mixamo.com).",
                        "OK");
                    return;
                }
                string path   = AssetDatabase.GUIDToAssetPath(guids[0]);
                var    prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                avatar        = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                avatar.name   = "Avatar_MonoArm";
                Debug.Log($"[SceneBuilder] Instantiated {path} as avatar.");
            }

            // Reset transform to the origin, facing the camera. The avatar may carry
            // a stale position/rotation override from a former multi-avatar layout
            // (e.g. one slot in a side-by-side comparison row), which would otherwise
            // leave it outside the camera's view after collapsing to a single avatar.
            Undo.RecordObject(avatar.transform, "Build MonoArm Scene");
            avatar.transform.position = Vector3.zero;
            avatar.transform.rotation = Quaternion.identity;

            // Check humanoid rig
            var animator = avatar.GetComponentInChildren<Animator>();
            if (animator == null || !animator.isHuman)
            {
                Undo.CollapseUndoOperations(undoGroup);
                EditorUtility.DisplayDialog("MonoArm — Humanoid Required",
                    $"'{avatar.name}' is not configured as a Humanoid.\n\n" +
                    "Select the FBX in the Project window → Inspector → Rig → " +
                    "Animation Type = Humanoid → Apply, then run Build Scene again.",
                    "OK");
                return;
            }

            // Remove any stale components from a previous build
            RemoveComponent<AvatarMuscleController>(avatar);
            RemoveComponent<PoseDebugUI>(avatar);

            // Add AvatarMuscleController to the avatar
            var ctrl = Undo.AddComponent<AvatarMuscleController>(avatar);
            EditorUtility.SetDirty(ctrl);

            // Create PoseManager GameObject
            var poseManagerGO = new GameObject("PoseManager");
            Undo.RegisterCreatedObjectUndo(poseManagerGO, "Build MonoArm Scene");

            var receiver   = poseManagerGO.AddComponent<UdpAngleReceiver>();
            var manager    = poseManagerGO.AddComponent<MonoArmManager>();
            var debugUI    = avatar.AddComponent<PoseDebugUI>();   // parented to avatar

            receiver.listenPort         = 9000;
            manager.receiver            = receiver;
            manager.avatarController    = ctrl;
            debugUI.receiver            = receiver;

            // Camera
            var cam = Camera.main;
            if (cam != null)
            {
                cam.transform.position = new Vector3(0f, 1.2f, -3.5f);
                cam.transform.rotation = Quaternion.Euler(5f, 0f, 0f);
            }

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Undo.CollapseUndoOperations(undoGroup);

            EditorUtility.DisplayDialog("MonoArm — Scene Built",
                $"Scene collapsed to a single humanoid and wired up.\n\n" +
                $"Avatar:  {avatar.name}\n" +
                $"UDP port: {receiver.listenPort}\n\n" +
                "Next steps:\n" +
                "1. Save the scene (Ctrl+S)\n" +
                "2. Run the Python pipeline:\n" +
                "   python -m scripts.run_demo\n" +
                "   -- or for testing without camera --\n" +
                "   python -m scripts.data_generator --mode sinusoidal\n" +
                "3. Press Play in Unity.",
                "OK");
        }

        // ── Collapse comparison scene → single avatar ───────────────────────

        /// <summary>
        /// Destroys everything left over from a multi-avatar comparison scene
        /// (extra humanoid avatars, "Label_*" world-space overlays, root
        /// Canvas objects) so exactly one humanoid avatar remains for the
        /// live UDP-driven MonoArm demo.
        /// </summary>
        static void CollapseToSingleAvatar()
        {
            // Keep exactly one humanoid Animator; destroy the rest.
            var humanoids = Object.FindObjectsOfType<Animator>(true)
                .Where(a => a != null && a.isHuman && a.gameObject.scene.IsValid())
                .Select(a => a.gameObject)
                .Distinct()
                .ToList();

            if (humanoids.Count > 1)
            {
                GameObject keep =
                    humanoids.FirstOrDefault(g => g.name == "Avatar_MediaPipe") ??
                    humanoids.FirstOrDefault(g => g.GetComponent<AvatarMuscleController>() != null) ??
                    humanoids[0];

                foreach (var g in humanoids)
                {
                    if (g == keep) continue;
                    Debug.Log($"[SceneBuilder] Removing extra avatar '{g.name}' (collapsing to single humanoid).");
                    Undo.DestroyObjectImmediate(g);
                }

                keep.name = "Avatar_MonoArm";
            }
            else if (humanoids.Count == 1)
            {
                humanoids[0].name = "Avatar_MonoArm";
            }

            // Destroy comparison-view overlays: root Canvas objects and any
            // "Label_*" objects (per-framework text overlays), plus stray
            // duplicates of "New Text".
            foreach (var c in Object.FindObjectsOfType<Canvas>(true))
            {
                if (c == null || !c.gameObject.scene.IsValid()) continue;
                if (c.transform.parent != null) continue; // leave avatar-parented UI (e.g. PoseDebugUI) alone
                Debug.Log($"[SceneBuilder] Removing comparison Canvas '{c.gameObject.name}'.");
                Undo.DestroyObjectImmediate(c.gameObject);
            }

            foreach (var t in Object.FindObjectsOfType<Transform>(true))
            {
                if (t == null || t.parent != null) continue; // root objects only
                if (!t.gameObject.scene.IsValid()) continue;
                if (t.name.StartsWith("Label_") || t.name == "New Text")
                {
                    Debug.Log($"[SceneBuilder] Removing comparison overlay '{t.name}'.");
                    Undo.DestroyObjectImmediate(t.gameObject);
                }
            }
        }

        static void DestroyAllNamed(string name)
        {
            GameObject go;
            while ((go = GameObject.Find(name)) != null)
                Undo.DestroyObjectImmediate(go);
        }

        // ── Diagnose ─────────────────────────────────────────────────────────

        [MenuItem("MonoArm/Diagnose Scene")]
        static void Diagnose()
        {
            var sb = new System.Text.StringBuilder();

            var receivers = Object.FindObjectsOfType<UdpAngleReceiver>();
            sb.AppendLine($"UdpAngleReceiver count: {receivers.Length}  (should be 1)");
            foreach (var r in receivers)
                sb.AppendLine($"  → on '{r.gameObject.name}' port={r.listenPort}");

            sb.AppendLine();

            var managers = Object.FindObjectsOfType<MonoArmManager>();
            sb.AppendLine($"MonoArmManager count: {managers.Length}  (should be 1)");
            foreach (var m in managers)
            {
                sb.AppendLine($"  receiver linked:   {m.receiver != null}");
                sb.AppendLine($"  controller linked: {m.avatarController != null}");
            }

            sb.AppendLine();

            var ctrls = Object.FindObjectsOfType<AvatarMuscleController>();
            sb.AppendLine($"AvatarMuscleController count: {ctrls.Length}  (should be 1)");
            foreach (var c in ctrls)
                sb.AppendLine($"  → on '{c.gameObject.name}'  enabled={c.enabled}");

            Debug.Log("[MonoArm Diagnose]\n" + sb);
            EditorUtility.DisplayDialog("MonoArm Scene Diagnosis", sb.ToString(), "OK");
        }

        // ── Undo ──────────────────────────────────────────────────────────────

        [MenuItem("MonoArm/Undo Last Build")]
        static void UndoBuild()
        {
            Undo.PerformUndo();
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        }

        // ── Helpers ───────────────────────────────────────────────────────────

        static GameObject FindHumanoidInScene()
        {
            foreach (var anim in Object.FindObjectsOfType<Animator>())
                if (anim.isHuman && anim.gameObject.scene.IsValid())
                    return anim.gameObject;
            return null;
        }

        static void RemoveComponent<T>(GameObject go) where T : Component
        {
            var c = go.GetComponent<T>();
            if (c != null) Undo.DestroyObjectImmediate(c);
        }
    }
}
#endif
