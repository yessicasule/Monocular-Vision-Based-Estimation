// PoseDebugUI.cs
// ===============
// World-space debug panel showing live joint angle values beneath the avatar.
// Parented to the avatar GameObject so it moves with it.
// Works with the MonoArm single-avatar architecture.

using System.Text;
using UnityEngine;
using UnityEngine.UI;

namespace MonoArm
{
    public class PoseDebugUI : MonoBehaviour
    {
        [Header("References")]
        public UdpAngleReceiver receiver;

        [Header("Display")]
        [Range(1f, 30f)]
        public float refreshHz = 10f;

        // Panel geometry (world units)
        const float PANEL_W     = 1.6f;
        const float PANEL_H     = 1.1f;
        const float PANEL_SCALE = 0.005f;
        const float PANEL_Y     = -1.15f;
        const float PANEL_Z     =  0.05f;

        // Colour theme
        static readonly Color ACCENT     = new(0.35f, 0.80f, 1.00f);   // MonoArm cyan
        static readonly Color BG_COLOR   = new(0.05f, 0.05f, 0.08f, 0.88f);
        static readonly Color TEXT_COLOR = Color.white;

        Text   _dataLabel;
        float  _nextRefresh;
        float  _lastDataTime;
        int    _lastSeenPacketCount;

        readonly StringBuilder _sb = new(256);

        void Awake()
        {
            if (receiver == null)
                receiver = FindFirstObjectByType<UdpAngleReceiver>();

            BuildPanel();
        }

        void Update()
        {
            if (receiver == null) return;

            // receiver.HasData stays true forever once the first packet arrives, so
            // gating on it alone would count every Unity frame as a new packet and
            // never let _lastDataTime go stale. Compare against receiver.PacketCount
            // (incremented only when a genuine new packet is consumed) instead.
            if (receiver.PacketCount != _lastSeenPacketCount)
            {
                _lastSeenPacketCount = receiver.PacketCount;
                _lastDataTime = Time.unscaledTime;
            }

            if (Time.unscaledTime < _nextRefresh) return;
            _nextRefresh = Time.unscaledTime + 1f / Mathf.Max(refreshHz, 1f);
            Refresh();
        }

        void Refresh()
        {
            if (_dataLabel == null) return;

            bool   waiting = !receiver.HasData;
            float  age     = Time.unscaledTime - _lastDataTime;
            string warn    = (!waiting && age > 0.5f) ? "  <color=#f55>⚠ stale</color>" : "";

            if (waiting)
            {
                _dataLabel.text = "<color=#555>Waiting for Python stream…\nRun scripts/run_demo.py</color>";
                return;
            }

            ArmAngles a = receiver.LatestAngles;
            _sb.Clear();

            string Row(string lbl, float val, string color) =>
                $"<color=#aaa>{lbl,-5}</color>  <color={color}><b>{val,+7:F1}°</b></color>\n";

            _sb.Append(Row("Flex",  a.shoulderFlexion,   "#64dc64"));  // green
            _sb.Append(Row("Abd",   a.shoulderAbduction,  "#64a0ff"));  // blue
            _sb.Append(Row("Rot",   a.shoulderRotation,   "#ffb450"));  // orange
            _sb.Append(Row("Elbow", a.elbowFlexion,       "#dc50dc"));  // purple
            _sb.Append($"\n<color=#444>pkts {receiver.PacketCount}   {age * 1000f:F0}ms{warn}</color>");

            _dataLabel.text = _sb.ToString();
        }

        void BuildPanel()
        {
            // Canvas parented to avatar root
            var cvGO = new GameObject("MonoArm_DebugPanel");
            cvGO.transform.SetParent(transform, false);
            cvGO.transform.localPosition = new Vector3(0f, PANEL_Y, PANEL_Z);
            cvGO.transform.localScale    = Vector3.one * PANEL_SCALE;

            var cv         = cvGO.AddComponent<Canvas>();
            cv.renderMode  = RenderMode.WorldSpace;
            cv.sortingOrder = 10;
            cvGO.AddComponent<CanvasScaler>();

            var rt         = cvGO.GetComponent<RectTransform>();
            rt.sizeDelta   = new Vector2(PANEL_W / PANEL_SCALE, PANEL_H / PANEL_SCALE);

            // Background
            var bg = new GameObject("BG");
            bg.transform.SetParent(cvGO.transform, false);
            var bgRT = bg.AddComponent<RectTransform>();
            bgRT.anchorMin  = Vector2.zero;
            bgRT.anchorMax  = Vector2.one;
            bgRT.offsetMin  = Vector2.zero;
            bgRT.offsetMax  = Vector2.zero;
            bg.AddComponent<Image>().color = BG_COLOR;

            // Accent bar
            var bar = new GameObject("AccentBar");
            bar.transform.SetParent(cvGO.transform, false);
            var barRT = bar.AddComponent<RectTransform>();
            barRT.anchorMin  = new Vector2(0f, 1f);
            barRT.anchorMax  = new Vector2(1f, 1f);
            barRT.pivot      = new Vector2(0.5f, 1f);
            barRT.offsetMin  = new Vector2(0f, -30f);
            barRT.offsetMax  = new Vector2(0f,   0f);
            bar.AddComponent<Image>().color = ACCENT;

            // Title label in accent bar
            var titleGO  = new GameObject("Title");
            titleGO.transform.SetParent(cvGO.transform, false);
            var titleRT  = titleGO.AddComponent<RectTransform>();
            titleRT.anchorMin = new Vector2(0f, 1f);
            titleRT.anchorMax = new Vector2(1f, 1f);
            titleRT.pivot     = new Vector2(0.5f, 1f);
            titleRT.offsetMin = new Vector2(8f, -30f);
            titleRT.offsetMax = new Vector2(-8f, -2f);
            var titleTxt      = titleGO.AddComponent<Text>();
            titleTxt.text      = "MonoArm  |  Right Arm";
            titleTxt.font      = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            titleTxt.fontSize  = 22;
            titleTxt.fontStyle = FontStyle.Bold;
            titleTxt.color     = new Color(0.05f, 0.05f, 0.05f);
            titleTxt.alignment = TextAnchor.MiddleCenter;

            // Data text
            var dataGO  = new GameObject("DataText");
            dataGO.transform.SetParent(cvGO.transform, false);
            var dataRT  = dataGO.AddComponent<RectTransform>();
            dataRT.anchorMin  = new Vector2(0f, 0f);
            dataRT.anchorMax  = new Vector2(1f, 1f);
            dataRT.offsetMin  = new Vector2(14f,  10f);
            dataRT.offsetMax  = new Vector2(-14f, -36f);

            _dataLabel              = dataGO.AddComponent<Text>();
            _dataLabel.font         = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            _dataLabel.fontSize     = 20;
            _dataLabel.lineSpacing  = 1.3f;
            _dataLabel.color        = TEXT_COLOR;
            _dataLabel.alignment    = TextAnchor.UpperLeft;
            _dataLabel.supportRichText      = true;
            _dataLabel.horizontalOverflow   = HorizontalWrapMode.Overflow;
            _dataLabel.verticalOverflow     = VerticalWrapMode.Overflow;
        }
    }
}
