// panels.jsx — Camera feed, audio beam viz, haptic log, status bar, info row, modal

const { useEffect, useRef, useState, useMemo } = React;

// ─────────────────────────────────────────────
// Camera feed (simulated tactical view + YOLO bboxes from threats in FOV)
// ─────────────────────────────────────────────
function CameraFeed({ threats, yoloDets, fps, yoloMs, temp, heading, liveCam, onLiveCamToggle, liveConnected }) {
  const canvasRef = useRef(null);
  const animRef   = useRef(0);
  const detsRef   = useRef(yoloDets || []);
  const liveRef   = useRef(liveConnected);
  useEffect(()=>{ detsRef.current = yoloDets || []; }, [yoloDets]);
  useEffect(()=>{ liveRef.current = liveConnected; }, [liveConnected]);
  const _feedHost = window.location.hostname || "raspberrypi.local";
  const _mjpegUrl = `http://${_feedHost}:8090/video_feed`;

  useEffect(()=>{
    const c = canvasRef.current; if(!c) return;
    function tick(){
      const dpr = window.devicePixelRatio||1;
      const rect = c.getBoundingClientRect();
      const W = Math.max(10, rect.width), H = Math.max(10, rect.height);
      if(c.width !== Math.round(W*dpr)){ c.width = Math.round(W*dpr); c.height = Math.round(H*dpr); }
      const ctx = c.getContext("2d");
      ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.clearRect(0,0,W,H);

      const dets = detsRef.current;
      const isLive = liveRef.current;

      if(!isLive){
        // Offline state: dark background + centered text
        ctx.fillStyle="#0a0e0d"; ctx.fillRect(0,0,W,H);
        ctx.fillStyle="rgba(32,201,151,0.35)";
        ctx.font="11px 'IBM Plex Mono',monospace";
        ctx.textAlign="center"; ctx.textBaseline="middle";
        ctx.fillText("NO FEED — PIPELINE OFFLINE",W/2,H/2);
        ctx.textAlign="left"; ctx.textBaseline="alphabetic";
        animRef.current = requestAnimationFrame(tick);
        return;
      }

      // Draw exact YOLO bboxes mapped from normalized coords to canvas size
      const URGENCY_COL = {critical:"#E24B4A", high:"#F59E0B", medium:"#20C997", low:"rgba(32,201,151,0.5)"};
      ctx.font = "bold 10px 'IBM Plex Mono',monospace";

      dets.forEach(d=>{
        const [x1n,y1n,x2n,y2n] = d.bbox_n;
        const bx=x1n*W, by=y1n*H, bw=(x2n-x1n)*W, bh=(y2n-y1n)*H;
        const col = URGENCY_COL[d.urgency] || "#20C997";
        const brk = Math.min(16, Math.max(6, bw*0.22));

        // Corner brackets
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.8;
        [[bx,by,1,1],[bx+bw,by,-1,1],[bx,by+bh,1,-1],[bx+bw,by+bh,-1,-1]].forEach(([x,y,sx,sy])=>{
          ctx.beginPath();
          ctx.moveTo(x+brk*sx,y); ctx.lineTo(x,y); ctx.lineTo(x,y+brk*sy);
          ctx.stroke();
        });

        // Label pill (top-left of box)
        const lbl = `${d.label}  ${Math.round(d.conf*100)}%`;
        const tw = ctx.measureText(lbl).width;
        ctx.fillStyle="rgba(0,0,0,0.72)";
        ctx.fillRect(bx, by-15, tw+10, 14);
        ctx.fillStyle=col;
        ctx.fillText(lbl, bx+5, by-4);

        // Distance tag (bottom-right)
        if(d.dist > 0){
          const dt = `${d.dist}m`;
          const dw = ctx.measureText(dt).width;
          ctx.fillStyle="rgba(0,0,0,0.72)";
          ctx.fillRect(bx+bw-dw-8, by+bh+1, dw+8, 13);
          ctx.fillStyle=col;
          ctx.fillText(dt, bx+bw-dw-3, by+bh+11);
        }
      });

      animRef.current = requestAnimationFrame(tick);
    }
    animRef.current = requestAnimationFrame(tick);
    return ()=>cancelAnimationFrame(animRef.current);
  },[]);

  const latClass = yoloMs<100?"ok":yoloMs<200?"warn":"err";

  return (
    <div className="cell tactical">
      <div className="panel-hd">
        <div className="ttl"><span className="tag">▸</span>CAMERA / YOLO v8-N</div>
        <div className="meta">
          <span>·</span>
          <button
            onClick={()=>onLiveCamToggle && onLiveCamToggle(!liveCam)}
            style={{
              background:"transparent",
              border:".5px solid "+(liveCam?"var(--teal)":"var(--div-hi)"),
              color: liveCam?"var(--teal)":"var(--tx-2)",
              fontFamily:"var(--mono)",fontSize:9,letterSpacing:".18em",
              padding:"2px 7px",cursor:"pointer",borderRadius:2
            }}>
            {liveCam ? "● OP-CAM" : "○ OP-CAM"}
          </button>
          <span>·</span>
          <span className={liveConnected?"live":""} style={{color:liveConnected?"var(--ac)":"var(--tx-3)"}}>
            {liveConnected?"● LIVE":"○ OFFLINE"}
          </span>
        </div>
      </div>
      {/* Video container: MJPEG behind exact-bbox canvas overlay */}
      <div className="cam" style={{margin:"10px", position:"relative"}}>
        <div style={{position:"relative",width:"100%",aspectRatio:"16/10",background:"#000",overflow:"hidden",borderRadius:3}}>
          {liveConnected && (
            <img src={_mjpegUrl} style={{
              position:"absolute",inset:0,width:"100%",height:"100%",
              objectFit:"fill", zIndex:0,
            }} alt="" />
          )}
          <canvas ref={canvasRef} style={{
            position:"absolute",inset:0,width:"100%",height:"100%",zIndex:1,
          }} />
          {/* HUD overlay */}
          <div style={{position:"absolute",top:6,left:8,display:"flex",gap:8,zIndex:2,fontFamily:"var(--mono)",fontSize:9,color:"var(--ac)"}}>
            <span>{fps}fps</span>
            <span style={{color:"#444"}}>|</span>
            <span className={"lat "+latClass}>{yoloMs}ms</span>
            <span style={{color:"#444"}}>|</span>
            <span>{temp}°C</span>
          </div>
          <div style={{position:"absolute",top:6,right:8,zIndex:2,fontFamily:"var(--mono)",fontSize:9,color:"var(--red)",animation:"blink 1.2s infinite"}}>● REC</div>
          <div style={{position:"absolute",bottom:5,left:0,right:0,display:"flex",justifyContent:"space-between",padding:"0 8px",zIndex:2,fontFamily:"var(--mono)",fontSize:8,color:"rgba(255,255,255,0.35)"}}>
            <span>CAM-0</span>
            <span>{liveConnected?`${(yoloDets||[]).length} DETS`:"OFFLINE"}</span>
            <span>YOLO·n</span>
          </div>
        </div>
        <LiveCamPiP
          enabled={!!liveCam}
          onClose={()=>onLiveCamToggle && onLiveCamToggle(false)}
          onRequestEnable={()=>onLiveCamToggle && onLiveCamToggle(true)}
        />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Audio Beam Visualizer (bars + polar)
// ─────────────────────────────────────────────
function AudioBeams({ beams, mode }) {
  const polarRef = useRef(null);
  useEffect(()=>{
    if(mode!=="polar") return;
    const c = polarRef.current; if(!c) return;
    const dpr = window.devicePixelRatio||1;
    const W = 220, H = 140;
    if(c.width!==W*dpr){ c.width=W*dpr; c.height=H*dpr; c.style.width=W+"px"; c.style.height=H+"px"; }
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,W,H);
    const cx=W/2, cy=H*0.7, R=Math.min(W*0.4, H*0.6);
    // grid
    ctx.strokeStyle="rgba(255,255,255,0.06)"; ctx.lineWidth=0.5;
    [0.33,0.66,1].forEach(f=>{ ctx.beginPath(); ctx.arc(cx,cy,R*f,Math.PI,0); ctx.stroke(); });
    // axes
    ctx.strokeStyle="rgba(255,255,255,0.05)";
    [180,225,270,315,360].forEach(d=>{
      const a = d*Math.PI/180;
      ctx.beginPath(); ctx.moveTo(cx,cy);
      ctx.lineTo(cx+Math.cos(a)*R, cy+Math.sin(a)*R); ctx.stroke();
    });
    // 4 beams: front=0(up), right=90, back=180(down), left=270
    const bs = [
      ["FRONT", beams.front, -90],
      ["RIGHT", beams.right, 0],
      ["LEFT",  beams.left, -180],
      ["BACK",  beams.back, 90],
    ];
    bs.forEach(([nm,e,deg])=>{
      const a = deg*Math.PI/180;
      const len = R*e;
      const hot = e>0.3;
      ctx.strokeStyle = hot ? "#20C997" : "rgba(120,140,135,0.7)";
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx+Math.cos(a)*len, cy+Math.sin(a)*len);
      ctx.stroke();
      // tip dot
      ctx.fillStyle = hot ? "#00e5ff" : "rgba(180,200,195,0.6)";
      ctx.beginPath();
      ctx.arc(cx+Math.cos(a)*len, cy+Math.sin(a)*len, 3,0,Math.PI*2);
      ctx.fill();
      // label
      ctx.fillStyle = "rgba(170,170,170,0.7)";
      ctx.font = "9px 'IBM Plex Mono',monospace";
      ctx.textAlign="center";
      ctx.fillText(nm, cx+Math.cos(a)*(R+10), cy+Math.sin(a)*(R+10)+3);
    });
    // center
    ctx.fillStyle = "#20C997";
    ctx.beginPath(); ctx.arc(cx,cy,3,0,Math.PI*2); ctx.fill();
  },[beams, mode]);

  return (
    <div className="cell tactical">
      <div className="panel-hd">
        <div className="ttl"><span className="tag">▸</span>AUDIO BEAMFORM</div>
        <div className="meta">
          <span>INMP441 ×4</span>
          <span>·</span>
          <span className="live">20HZ</span>
        </div>
      </div>
      {mode==="polar" ? (
        <div className="beam-polar">
          <canvas ref={polarRef} />
        </div>
      ) : (
        <div className="beams">
          {[
            ["FRONT", beams.front],
            ["RIGHT", beams.right],
            ["BACK",  beams.back],
            ["LEFT",  beams.left],
          ].map(([nm,e])=>(
            <div key={nm} className="row">
              <div className="nm">{nm}</div>
              <div className="bar"><div className={"fill "+(e>0.4?"hot":"")} style={{width: (e*100).toFixed(0)+"%"}}/></div>
              <div className="pct">{Math.round(e*100)}%</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────
// Haptic Event Log
// ─────────────────────────────────────────────
function HapticLog({ events, expanded, setExpanded, onAck }) {
  const listRef = useRef(null);
  const stickRef = useRef(true);
  useEffect(()=>{
    const el = listRef.current; if(!el) return;
    function onScroll(){
      const near = el.scrollTop < 8;
      stickRef.current = near;
    }
    el.addEventListener("scroll", onScroll);
    return ()=>el.removeEventListener("scroll", onScroll);
  },[]);
  useEffect(()=>{
    if(stickRef.current && listRef.current){
      listRef.current.scrollTop = 0;
    }
  },[events.length]);

  return (
    <div className="cell area-log tactical">
      <div className="panel-hd">
        <div className="ttl"><span className="tag">▸</span>HAPTIC EVENT LOG</div>
        <div className="meta">
          <span>{events.length} EVT</span>
          <span>·</span>
          <span className="live">RT</span>
        </div>
      </div>
      <div className="log-list" ref={listRef}>
        {events.map(ev=>(
          <div key={ev.id}
            className={"log-row "+(ev.urgency==="critical"?"crit ":"")+(expanded===ev.id?"expanded":"")}
            onClick={()=>setExpanded(expanded===ev.id?null:ev.id)}>
            <div className="ln1">
              <span className="time">{hhmmss(ev.bornAt)}</span>
              <span className="dir">{arrowFor(ev.angle)} {compass(ev.angle)}</span>
              <span className={"src "+ev.source.toLowerCase()}>{ev.source}</span>
            </div>
            <div className="pat">{patternFor(ev.urgency)}</div>
            <div className="sub">{ev.distance.toFixed(1)}m · {ev.confidence}% · {ev.label.toLowerCase()}</div>
            <div className="det">
              <div className="kv"><span>id</span><span>{ev.id}</span></div>
              <div className="kv"><span>velocity</span><span>{ev.velocity>=0?"+":""}{ev.velocity.toFixed(2)} m/s</span></div>
              <div className="kv"><span>eta</span><span>{ev.eta.toFixed(1)}s</span></div>
              <div className="kv"><span>audio</span><span>{ev.audioClass}</span></div>
              <div className="kv"><span>visual</span><span>{ev.visualClass}</span></div>
              <div className="summary">{summarize(ev)}</div>
            </div>
          </div>
        ))}
      </div>
      <div className="log-foot">
        <span>{events.filter(e=>e.urgency==="critical").length} CRIT · {events.filter(e=>e.urgency==="high").length} HIGH</span>
        <button onClick={onAck}>ACK ALL</button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Info row — closest threat detail strip
// ─────────────────────────────────────────────
function InfoRow({ threats, heading }) {
  const closest = threats.length ? [...threats].sort((a,b)=>a.distance-b.distance)[0] : null;
  const critCount = threats.filter(t=>t.urgency==="critical").length;
  return (
    <div className="cell area-info tactical">
      <div className="panel-hd">
        <div className="ttl"><span className="tag">▸</span>NEAREST THREAT · IMU FUSION</div>
        <div className="meta">
          <span>{threats.length} ACTIVE</span>
          <span>·</span>
          <span className={critCount?"live":""} style={critCount?{color:"#E24B4A"}:undefined}>
            {critCount?"⚠ CRITICAL":"NOMINAL"}
          </span>
        </div>
      </div>
      <div className="info">
        <div className="col">
          <div className="k">Subject</div>
          <div className="v">{closest ? closest.icon+" "+closest.label : "—"}</div>
          <div className="ind">{closest ? `${closest.source} · ${closest.audioClass}` : "no targets in range"}</div>
        </div>
        <div className="col">
          <div className="k">Bearing / Range</div>
          <div className="v">
            {closest ? `${compass(closest.angle)}` : "—"}
            <span className="u">{closest ? `${closest.distance.toFixed(1)} m` : ""}</span>
          </div>
          <div className={"ind "+(closest && closest.urgency==="critical"?"red":"teal")}>
            {closest ? `${Math.round(closest.angle)}° abs · ${Math.round(((closest.angle-heading+540)%360)-180)}° rel` : ""}
          </div>
        </div>
        <div className="col">
          <div className="k">Closure / Conf</div>
          <div className="v">
            {closest ? `${closest.velocity>=0?"+":""}${closest.velocity.toFixed(1)}` : "—"}
            <span className="u">m/s</span>
          </div>
          <div className="bars">
            {Array.from({length:10}).map((_,i)=>(
              <span key={i} className={closest && i < Math.round(closest.confidence/10) ? "on":""} style={{height: (4+i*1.4)+"px"}}/>
            ))}
          </div>
          <div className="ind">{closest ? `${closest.confidence}% conf · ETA ${closest.eta.toFixed(1)}s` : ""}</div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Status bar (bottom)
// ─────────────────────────────────────────────
function StatusBar({ stat, beams, signal, muted }) {
  const gemmaCls = stat.gemma<3?"green":stat.gemma<5?"yellow":"red";
  const pipeCls = stat.pipeline>8?"green":stat.pipeline>6?"yellow":"red";
  const yoloCls = stat.yolo<100?"green":stat.yolo<150?"yellow":"red";
  const tempCls = stat.temp<50?"green":stat.temp<60?"yellow":"red";
  const sigCls = signal===4?"green":signal===3?"yellow":signal===2?"yellow":"red";

  // mic bar heights (24px max)
  const mics = [beams.front,beams.right,beams.back,beams.left];

  return (
    <div className="statusbar">
      <div className="sb-item" title="Wearable identity">
        <span className="lbl">UNIT</span>
        <span className="val teal">WS-04F</span>
      </div>
      <div className="sb-item">
        <span className="lbl">GEMMA</span>
        <span className={"val "+gemmaCls}>{stat.gemma.toFixed(1)}s</span>
      </div>
      <div className="sb-item">
        <span className="lbl">PIPELINE</span>
        <span className={"val "+pipeCls}>{stat.pipeline.toFixed(1)}fps</span>
      </div>
      <div className="sb-item">
        <span className="lbl">YOLO</span>
        <span className={"val "+yoloCls}>{stat.yolo}ms</span>
      </div>
      <div className="sb-item">
        <span className="lbl">MICS</span>
        <span className="mics">
          {mics.map((e,i)=>(<span key={i} className={e>0.2?"on":""} style={{height: (4+e*20).toFixed(0)+"px"}}/>))}
        </span>
      </div>
      <div className="sb-item">
        <span className="lbl">CPU</span>
        <span className={"val "+tempCls}>{stat.temp}°C</span>
      </div>
      <div className="sb-item">
        <span className="lbl">BATT</span>
        <span className="val">{stat.batt}%</span>
      </div>
      <div className="sb-item">
        <span className="lbl">HAPTIC</span>
        <span className={"val "+(muted?"red":"teal")}>{muted?"MUTED":"ARMED"}</span>
      </div>
      <div className="sb-item">
        <span className="lbl">SIGNAL</span>
        <span className="sig">
          {[1,2,3,4].map(i=>(
            <span key={i} className={"lvl "+(i<=signal?"on":"")}>◆</span>
          ))}
        </span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Threat modal
// ─────────────────────────────────────────────
function ThreatModal({ threat, onClose, onAck, onTrack }) {
  if(!threat) return null;
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={e=>e.stopPropagation()}>
        <div className="modal-hd">
          <div className="ttl">▸ THREAT · {threat.label}</div>
          <div className="id">{threat.id} · {threat.source}</div>
        </div>
        <div className="modal-bd">
          <div className="mb">
            <div className="k">Distance</div>
            <div className="v">{threat.distance.toFixed(2)}<span className="u">m</span></div>
          </div>
          <div className="mb">
            <div className="k">Bearing</div>
            <div className="v">{compass(threat.angle)} <span className="u">{Math.round(threat.angle)}°</span></div>
          </div>
          <div className="mb">
            <div className="k">Closure</div>
            <div className="v">{threat.velocity>=0?"+":""}{threat.velocity.toFixed(2)}<span className="u">m/s</span></div>
          </div>
          <div className="mb">
            <div className="k">ETA</div>
            <div className="v">{threat.eta.toFixed(1)}<span className="u">s</span></div>
          </div>
          <div className="mb">
            <div className="k">Confidence</div>
            <div className="v">{threat.confidence}<span className="u">%</span></div>
          </div>
          <div className="mb">
            <div className="k">Urgency</div>
            <div className="v" style={{color:{critical:"#E24B4A",high:"#BA7517",medium:"#FFD54F",low:"#888"}[threat.urgency]}}>{threat.urgency.toUpperCase()}</div>
          </div>
          <div className="mb full">
            <div className="k">Multimodal summary</div>
            <div className="summary">
              <strong style={{color:"#cfcfcf"}}>AUDIO</strong> · {threat.audioClass}<br/>
              <strong style={{color:"#cfcfcf"}}>VISUAL</strong> · {threat.visualClass}<br/>
              <strong style={{color:"#cfcfcf"}}>FUSION</strong> · {summarize(threat)}
            </div>
          </div>
        </div>
        <div className="modal-ft">
          <button onClick={()=>{onTrack(threat); onClose();}}>TRACK</button>
          <button className="danger" onClick={()=>{onAck(threat); onClose();}}>MUTE / ACK</button>
          <button onClick={onClose}>DISMISS</button>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Live Camera PiP — actual webcam feed (operator verification cam)
// ─────────────────────────────────────────────
function LiveCamPiP({ enabled, onClose, onRequestEnable }) {
  const videoRef = useRef(null);
  const [state, setState] = useState("idle"); // idle | requesting | live | denied | error | unsupported
  const [err, setErr] = useState("");
  const streamRef = useRef(null);
  const [nowTs, setNowTs] = useState("");

  useEffect(()=>{
    if(!enabled){
      // stop stream
      if(streamRef.current){
        streamRef.current.getTracks().forEach(t=>t.stop());
        streamRef.current = null;
      }
      if(videoRef.current) videoRef.current.srcObject = null;
      setState("idle");
      return;
    }
    let cancelled = false;
    async function start(){
      if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
        setState("unsupported");
        return;
      }
      setState("requesting");
      try{
        const s = await navigator.mediaDevices.getUserMedia({
          video: { width:{ideal:480}, height:{ideal:360}, facingMode:"user" },
          audio: false
        });
        if(cancelled){ s.getTracks().forEach(t=>t.stop()); return; }
        streamRef.current = s;
        if(videoRef.current){
          videoRef.current.srcObject = s;
          videoRef.current.play().catch(()=>{});
        }
        setState("live");
      } catch(e){
        if(cancelled) return;
        const name = (e && e.name) || "";
        if(name==="NotAllowedError" || name==="PermissionDeniedError"){
          setState("denied"); setErr("permission denied");
        } else if(name==="NotFoundError" || name==="DevicesNotFoundError"){
          setState("error"); setErr("no camera detected");
        } else {
          setState("error"); setErr((e && e.message) || "stream failed");
        }
      }
    }
    start();
    return ()=>{
      cancelled = true;
      if(streamRef.current){
        streamRef.current.getTracks().forEach(t=>t.stop());
        streamRef.current = null;
      }
    };
  },[enabled]);

  // local timestamp tick when live
  useEffect(()=>{
    if(state!=="live") return;
    const id = setInterval(()=>{
      const d = new Date();
      const p = n=>String(n).padStart(2,"0");
      setNowTs(`${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`);
    }, 500);
    return ()=>clearInterval(id);
  },[state]);

  if(!enabled) return null;

  return (
    <div className="livecam">
      <video ref={videoRef} playsInline muted autoPlay/>
      <div className="lc-recticle" />
      <button className="lc-close" onClick={onClose} title="Close live cam">×</button>
      <div className="lc-hd">
        <span className={"nm "+(state==="live"?"":"off")}>
          {state==="live" ? "LIVE" : state==="requesting" ? "INIT" : "OFFLINE"}
        </span>
        <span style={{color:"rgba(255,255,255,.7)"}}>OP-CAM</span>
      </div>
      {state==="live" && (
        <div className="lc-ft">
          <span>{nowTs}</span>
          <span>{streamRef.current?.getVideoTracks?.()[0]?.getSettings?.()?.width || ""}×{streamRef.current?.getVideoTracks?.()[0]?.getSettings?.()?.height || ""}</span>
        </div>
      )}
      {(state==="requesting" || state==="idle") && (
        <div className="lc-state">
          <div className="icon">⟳</div>
          <div>REQUESTING<br/>OPERATOR CAM…</div>
        </div>
      )}
      {state==="denied" && (
        <div className="lc-state err">
          <div className="icon">⊘</div>
          <div>ACCESS DENIED<br/><span style={{color:"var(--tx-3)"}}>{err}</span></div>
          <button onClick={onRequestEnable}>RETRY</button>
        </div>
      )}
      {state==="error" && (
        <div className="lc-state err">
          <div className="icon">!</div>
          <div>CAM FAULT<br/><span style={{color:"var(--tx-3)"}}>{err}</span></div>
          <button onClick={onRequestEnable}>RETRY</button>
        </div>
      )}
      {state==="unsupported" && (
        <div className="lc-state err">
          <div className="icon">⊘</div>
          <div>WEBRTC<br/>UNAVAILABLE</div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────
// Pipeline Camera Feed — MJPEG stream from Pi with YOLO boxes
// ─────────────────────────────────────────────
function PipelineFeed({ liveConnected }) {
  const [loaded, setLoaded] = useState(false);
  const [err, setErr]       = useState(false);
  const imgRef = useRef(null);
  const _host = window.location.hostname || "raspberrypi.local";
  const url = `http://${_host}:8090/video_feed`;

  useEffect(() => {
    if (!liveConnected) { setLoaded(false); setErr(false); return; }
    if (imgRef.current) {
      imgRef.current.src = "";
      imgRef.current.src = url + "?t=" + Date.now();
    }
  }, [liveConnected]);

  return (
    <div className="panel pipeline-feed-panel">
      <div className="panel-hd">
        <span className="dot" style={{color: liveConnected && !err ? "var(--ac)" : "#E24B4A"}}>●</span>
        <span className="ttl">PIPELINE CAM · YOLO</span>
        <span style={{marginLeft:"auto", fontSize:9, letterSpacing:2, opacity:.55, color: liveConnected && !err ? "var(--ac)" : "#E24B4A"}}>
          {liveConnected && !err ? "LIVE" : "OFFLINE"}
        </span>
      </div>
      <div style={{position:"relative", background:"#000", borderRadius:3, overflow:"hidden", aspectRatio:"4/3"}}>
        {!loaded && !err && (
          <div style={{position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",color:"var(--ac)",fontSize:10,letterSpacing:2,opacity:.5}}>
            CONNECTING…
          </div>
        )}
        {err && (
          <div style={{position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",flexDirection:"column",gap:6,color:"#E24B4A",fontSize:10,letterSpacing:2}}>
            <span style={{fontSize:18}}>⊘</span><span>NO FEED</span>
          </div>
        )}
        <img
          ref={imgRef}
          src={liveConnected ? url : ""}
          onLoad={() => { setLoaded(true); setErr(false); }}
          onError={() => { setErr(true); setLoaded(false); }}
          style={{width:"100%",height:"100%",objectFit:"contain",display:loaded?"block":"none"}}
          alt="pipeline feed"
        />
        {/* tactical corners */}
        {[{top:4,left:4,borderTop:"1px solid",borderLeft:"1px solid"},{top:4,right:4,borderTop:"1px solid",borderRight:"1px solid"},{bottom:4,left:4,borderBottom:"1px solid",borderLeft:"1px solid"},{bottom:4,right:4,borderBottom:"1px solid",borderRight:"1px solid"}].map((s,i)=>(
          <div key={i} style={{position:"absolute",width:10,height:10,borderColor:"var(--ac)",opacity:.5,...s}}/>
        ))}
      </div>
    </div>
  );
}

Object.assign(window, { CameraFeed, AudioBeams, HapticLog, InfoRow, StatusBar, ThreatModal, LiveCamPiP, PipelineFeed });
