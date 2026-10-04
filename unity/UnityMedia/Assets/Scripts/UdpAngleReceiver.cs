// UdpAngleReceiver.cs
// ===================
// Receives arm joint angle packets from the MonoArm Python pipeline over UDP.
//
// Packet Format
// -------------
//   S,<shoulder_flex>,<shoulder_abd>,<shoulder_rot>,<elbow_flex>\n
//       Single-arm (right) pose from MediaPipe (calibrated + filtered on
//       the Python side). Drives the single humanoid avatar in the scene.
//
//   B,<r_flex>,<r_abd>,<r_rot>,<r_elbow>,<l_flex>,<l_abd>,<l_rot>,<l_elbow>\n
//       Bilateral pose (right side first). LatestAngles is kept mirroring
//       the right side for backward compatibility with single-avatar
//       scenes; LatestBilateralAngles carries both sides.
//
//   All values are in degrees.
//
// Threading Model
// ---------------
//   A background thread calls UdpClient.Receive() (blocking). When a packet
//   arrives it is parsed into a pending ArmAngles struct under a lock.
//   On the Unity main thread (Update()), the pending struct is copied into
//   the public LatestAngles property under the same lock. This two-stage
//   approach keeps the main thread non-blocking and avoids race conditions.
//
// Socket Options
// --------------
//   ReuseAddress is set so the OS releases the port immediately when the
//   receiver is destroyed, preventing "address already in use" errors
//   during rapid Enter/Exit Play mode cycles in the Unity Editor.

using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace MonoArm
{
    /// <summary>
    /// Anatomically labelled arm joint angles received from the Python pipeline.
    /// All values are in degrees.
    /// </summary>
    public struct ArmAngles
    {
        /// <summary>Shoulder flexion (+) / extension (−) in degrees.</summary>
        public float shoulderFlexion;

        /// <summary>Shoulder abduction (+) / adduction (−) in degrees.</summary>
        public float shoulderAbduction;

        /// <summary>
        /// Shoulder internal (+) / external (−) rotation in degrees.
        /// Reliable only when elbow is flexed ≥ 25°.
        /// </summary>
        public float shoulderRotation;

        /// <summary>Elbow flexion in degrees. 0° = fully extended.</summary>
        public float elbowFlexion;

        public override string ToString() =>
            $"Flex:{shoulderFlexion:F1}° Abd:{shoulderAbduction:F1}° " +
            $"Rot:{shoulderRotation:F1}° Elb:{elbowFlexion:F1}°";
    }

    /// <summary>Both arms' angles from one bilateral ('B,') packet.</summary>
    public struct BilateralArmAngles
    {
        public ArmAngles right;
        public ArmAngles left;

        public override string ToString() => $"R[{right}] L[{left}]";
    }

    /// <summary>
    /// Threaded UDP receiver that delivers ArmAngles to the Unity main thread.
    /// Attach to any persistent GameObject (e.g. a PoseManager empty object).
    /// Only one instance may bind the configured port at a time.
    /// </summary>
    public class UdpAngleReceiver : MonoBehaviour
    {
        [Header("Network")]
        [Tooltip("UDP port to listen on. Must match Python stream_hz port (default 9000).")]
        public int listenPort = 9000;

        // ── Public state ────────────────────────────────────────────────────
        /// <summary>
        /// Latest validated angles, updated each Unity frame. For a bilateral
        /// ('B,') packet this mirrors the right side, for backward
        /// compatibility with single-avatar scenes.
        /// </summary>
        public ArmAngles LatestAngles { get; private set; }

        /// <summary>Latest bilateral angles (both arms), updated each Unity frame.</summary>
        public BilateralArmAngles LatestBilateralAngles { get; private set; }

        /// <summary>True if the most recently received packet was a bilateral ('B,') packet.</summary>
        public bool IsBilateral { get; private set; }

        /// <summary>True once the first valid packet has been received.</summary>
        public bool HasData { get; private set; }

        /// <summary>Total number of valid packets received in this session.</summary>
        public int PacketCount { get; private set; }

        /// <summary>Elapsed seconds since the last valid packet arrived.</summary>
        public float TimeSinceLastPacket { get; private set; }

        // ── Private state ───────────────────────────────────────────────────
        UdpClient     _client;
        Thread        _thread;
        volatile bool _running;

        readonly object _lock = new();
        ArmAngles _pending;
        BilateralArmAngles _pendingBilateral;
        bool      _pendingIsBilateral;
        bool      _pendingReady;
        float     _lastPacketTime;

        // Parse-error throttling. A malformed sender streams malformed packets,
        // so an unguarded warning here fires once per packet — tens of times a
        // second, from a background thread, each one capturing a stack trace and
        // marshalling to the main thread. That alone is enough to stall scene
        // updates. Report the first failure, then at most one summary per
        // interval, and never lose the total count.
        const double ParseErrorLogIntervalSeconds = 5.0;
        int      _parseErrorCount;
        int      _parseErrorsAtLastLog;
        DateTime _lastParseErrorLogUtc = DateTime.MinValue;

        // Singleton guard — one receiver per scene
        static UdpAngleReceiver _instance;

        // ── Unity lifecycle ─────────────────────────────────────────────────

        void Awake()
        {
            if (_instance != null && _instance != this)
            {
                Debug.LogWarning(
                    $"[UdpAngleReceiver] Duplicate on '{gameObject.name}' destroyed. " +
                    $"Only one receiver may bind port {listenPort}.");
                Destroy(this);
                return;
            }
            _instance = this;
            StartReceiver();
        }

        void OnDestroy()
        {
            StopReceiver();
            if (_instance == this) _instance = null;
        }

        void Update()
        {
            TimeSinceLastPacket = Time.unscaledTime - _lastPacketTime;

            lock (_lock)
            {
                if (!_pendingReady) return;
                LatestAngles           = _pending;
                IsBilateral            = _pendingIsBilateral;
                if (_pendingIsBilateral)
                    LatestBilateralAngles = _pendingBilateral;
                _pendingReady  = false;
                if (!HasData)
                    Debug.Log($"[UdpAngleReceiver] First packet received on port {listenPort} — avatar is live.");
                HasData        = true;
                PacketCount++;
                _lastPacketTime = Time.unscaledTime;
            }
        }

        // ── Socket management ───────────────────────────────────────────────

        void StartReceiver()
        {
            try
            {
                var sock = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
                sock.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
                sock.Bind(new IPEndPoint(IPAddress.Any, listenPort));
                _client  = new UdpClient { Client = sock };
                _running = true;
                _thread  = new Thread(ReceiveLoop) { IsBackground = true, Name = "UdpReceiver" };
                _thread.Start();
                Debug.Log($"[UdpAngleReceiver] Listening on UDP port {listenPort}");
            }
            catch (SocketException ex)
            {
                Debug.LogError(
                    $"[UdpAngleReceiver] Cannot bind port {listenPort}: {ex.Message}\n" +
                    "Ensure no other script has UdpAngleReceiver and the port is not in use.");
            }
        }

        void StopReceiver()
        {
            _running = false;
            try { _client?.Close(); } catch { }
            _client = null;
            _thread?.Join(500);
            _thread = null;
        }

        // ── Background receive loop ─────────────────────────────────────────

        void ReceiveLoop()
        {
            var ep = new IPEndPoint(IPAddress.Any, listenPort);
            while (_running)
            {
                try
                {
                    byte[] data = _client.Receive(ref ep);
                    string line = Encoding.UTF8.GetString(data).Trim();

                    if (string.IsNullOrEmpty(line)) continue;

                    if (line.StartsWith("S,"))
                    {
                        // Parse: S,flex,abd,rot,elbow
                        if (!TryParsePacket(line.Substring(2), out ArmAngles angles)) continue;

                        lock (_lock)
                        {
                            _pending            = angles;
                            _pendingIsBilateral = false;
                            _pendingReady       = true;
                        }
                    }
                    else if (line.StartsWith("B,"))
                    {
                        // Parse: B,r_flex,r_abd,r_rot,r_elbow,l_flex,l_abd,l_rot,l_elbow
                        if (!TryParseBilateralPacket(line.Substring(2), out BilateralArmAngles bilateral)) continue;

                        lock (_lock)
                        {
                            _pending            = bilateral.right;
                            _pendingBilateral   = bilateral;
                            _pendingIsBilateral = true;
                            _pendingReady       = true;
                        }
                    }
                }
                catch (SocketException)  { /* socket closed — exit loop */ break; }
                catch (ObjectDisposedException) { break; }
                catch (Exception ex)
                {
                    ReportParseError(ex);
                }
            }
        }

        /// <summary>
        /// Record a parse failure, logging at most once per
        /// <see cref="ParseErrorLogIntervalSeconds"/>.
        /// </summary>
        /// <remarks>
        /// Called from the background receive thread. Everything before the
        /// interval check is a counter increment and a clock read, so the
        /// common case — a steady stream of malformed packets — costs no string
        /// formatting, no stack-trace capture, and no main-thread marshalling.
        /// Suppressed failures are counted and reported in the next summary, so
        /// throttling hides none of the diagnostic signal.
        ///
        /// DateTime.UtcNow is used rather than Time.unscaledTime because the
        /// Unity time API is main-thread only.
        /// </remarks>
        void ReportParseError(Exception ex)
        {
            _parseErrorCount++;

            DateTime now = DateTime.UtcNow;
            if ((now - _lastParseErrorLogUtc).TotalSeconds < ParseErrorLogIntervalSeconds)
                return;

            int suppressed = _parseErrorCount - _parseErrorsAtLastLog - 1;
            _lastParseErrorLogUtc  = now;
            _parseErrorsAtLastLog  = _parseErrorCount;

            string suffix = suppressed > 0
                ? $" ({suppressed} similar suppressed in the last "
                  + $"{ParseErrorLogIntervalSeconds:0}s; {_parseErrorCount} total)"
                : string.Empty;

            Debug.LogWarning($"[UdpAngleReceiver] Parse error: {ex.Message}{suffix}");
        }

        // ── Packet parser ───────────────────────────────────────────────────

        static bool TryParsePacket(string body, out ArmAngles a)
        {
            a = default;
            var  parts = body.Split(',');
            if   (parts.Length < 4) return false;

            var inv = System.Globalization.CultureInfo.InvariantCulture;
            var fl  = System.Globalization.NumberStyles.Float;

            if (!float.TryParse(parts[0], fl, inv, out float flex))  return false;
            if (!float.TryParse(parts[1], fl, inv, out float abd))   return false;
            if (!float.TryParse(parts[2], fl, inv, out float rot))   return false;
            if (!float.TryParse(parts[3], fl, inv, out float elbow)) return false;

            a = new ArmAngles
            {
                shoulderFlexion   = flex,
                shoulderAbduction = abd,
                shoulderRotation  = rot,
                elbowFlexion      = elbow,
            };
            return true;
        }

        static bool TryParseBilateralPacket(string body, out BilateralArmAngles a)
        {
            a = default;
            var parts = body.Split(',');
            if (parts.Length < 8) return false;

            if (!TryParsePacket(string.Join(",", parts, 0, 4), out ArmAngles right)) return false;
            if (!TryParsePacket(string.Join(",", parts, 4, 4), out ArmAngles left))  return false;

            a = new BilateralArmAngles { right = right, left = left };
            return true;
        }
    }
}
