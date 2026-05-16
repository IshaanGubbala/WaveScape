// app.jsx — WaveScape main orchestrator (v1)

const { useState, useEffect, useRef, useCallback } = React;

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "beamMode": "bars",
  "sweep": true,
  "scanlines": true,
  "spawnRate": "normal",
  "accent": "teal",
  "theme": "dark",
  "liveCam": false
}/*EDITMODE-END*/;

const ACCENTS = {
  teal:   { name:"teal",   hex:"#20C997", cyan:"#00e5ff", dim:"#147a5d",
            lightHex:"#0a8a64", lightCyan:"#0a8aa6", lightDim:"#6cb89e" },
  amber:  { name:"amber",  hex:"#FFB020", cyan:"#FFD54F", dim:"#7a5a14",
            lightHex:"#a36400", lightCyan:"#b58c00", lightDim:"#d4a85a" },
  cyan:   { name:"cyan",   hex:"#00e5ff", cyan:"#20C997", dim:"#0a6a7a",
            lightHex:"#0a8aa6", lightCyan:"#0a8a64", lightDim:"#6ca8b8" },
  rose:   { name:"rose",   hex:"#E24B7A", cyan:"#FF9DBF", dim:"#7a1842",
            lightHex:"#b53560", lightCyan:"#c87693", lightDim:"#d99cb3" },
};

function applyAccent(name, theme){
  const a = ACCENTS[name] || ACCENTS.teal;
  const r = document.documentElement.style;
  if(theme==="light"){
    r.setProperty("--teal", a.lightHex || a.hex);
    r.setProperty("--cyan", a.lightCyan || a.cyan);
    r.setProperty("--teal-dim", a.lightDim || a.dim);
  } else {
    r.setProperty("--teal", a.hex);
    r.setProperty("--cyan", a.cyan);
    r.setProperty("--teal-dim", a.dim);
  }
}

function applyTheme(theme){
  document.documentElement.setAttribute("data-theme", theme==="light"?"light":"dark");
}

function App(){
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  useEffect(()=>{ applyTheme(t.theme); },[t.theme]);
  useEffect(()=>{ applyAccent(t.accent, t.theme); },[t.accent, t.theme]);

  const [threats, setThreats] = useState([]);
  const [events, setEvents] = useState([]);
  const [expandedEv, setExpandedEv] = useState(null);
  const [beams, setBeams] = useState({front:.12, right:.05, back:.08, left:.18});
  const [heading, setHeading] = useState(0);
  const [stat, setStat] = useState({gemma:2.6, pipeline:9.2, yolo:78, temp:42, batt:84});
  const [signal, setSignal] = useState(4);
  const [muted, setMuted] = useState(false);
  const [paused, setPaused] = useState(false);
  const [now, setNow] = useState(new Date());
  const [modalThreat, setModalThreat] = useState(null);
  const [showHelp, setShowHelp] = useState(false);
  const [showSb, setShowSb] = useState(true);
  const [liveConnected, setLiveConnected] = useState(false);
  const [yoloDets, setYoloDets] = useState([]);

  // Live SSE from pipeline (port 8090). Falls back to sim when disconnected.
  useEffect(()=>{
    const host = window.location.hostname || "raspberrypi.local";
    const url = `http://${host}:8090/stream`;
    let es;
    let retry;
    function connect(){
      es = new EventSource(url);
      es.onopen = ()=>{ setLiveConnected(true); setSignal(4); };
      es.onmessage = (e)=>{
        try{
          const d = JSON.parse(e.data);
          if(d.error) return;
          if(d.threats   !== undefined) setThreats(d.threats.map(th=>({...th, bornAt: th.bornAt || Date.now(), ttl: th.ttl || 3000, coasting: th.coasting||false, crossing: th.crossing||false, predicted: th.predicted||[], _rx: Date.now()})));
          if(d.events    !== undefined) setEvents(d.events);
          if(d.beams     !== undefined) setBeams(b=>({...b, ...d.beams}));
          if(d.heading   !== undefined) setHeading(d.heading);
          if(d.stat      !== undefined) setStat(s=>({...s, ...d.stat}));
          if(d.yolo_dets !== undefined) setYoloDets(d.yolo_dets);
        } catch(_){}
      };
      es.onerror = ()=>{
        setLiveConnected(false);
        setSignal(1);
        es.close();
        retry = setTimeout(connect, 3000);
      };
    }
    connect();
    return ()=>{ if(es) es.close(); if(retry) clearTimeout(retry); };
  },[]);

  // Clock
  useEffect(()=>{
    const id = setInterval(()=>setNow(new Date()), 250);
    return ()=>clearInterval(id);
  },[]);

  // Threat lifecycle — spawn, age out (sim only when not live)
  useEffect(()=>{
    if(paused || liveConnected) return;
    const spawnInterval = t.spawnRate==="quiet" ? 3200 : t.spawnRate==="storm" ? 700 : 1500;
    const spawn = setInterval(()=>{
      setThreats(prev => {
        if(prev.length >= (t.spawnRate==="storm"?14:8)) return prev;
        const tr = newThreat();
        // Bias toward front-cone (slightly)
        return [...prev, tr];
      });
    }, spawnInterval);
    return ()=>clearInterval(spawn);
  },[paused, t.spawnRate]);

  // Age & move threats (sim only)
  useEffect(()=>{
    if(paused || liveConnected) return;
    const id = setInterval(()=>{
      const n = Date.now();
      setThreats(prev => prev
        .map(th=>{
          // closure update
          const dt = 0.12;
          const newDist = Math.max(0.4, th.distance - th.velocity*dt);
          const newUrg = urgencyFromDist(newDist);
          // small angular drift
          const newAng = (th.angle + (Math.random()*2-1)*0.6 + 360)%360;
          return {...th, distance:newDist, angle:newAng, urgency:newUrg};
        })
        .filter(th => (n - th.bornAt) < th.ttl)
      );
    }, 120);
    return ()=>clearInterval(id);
  },[paused]);

  // Whenever a new threat appears, add to event log (sim only — live server sends events directly)
  const lastIdsRef = useRef(new Set());
  useEffect(()=>{
    if(liveConnected) return;
    const curr = new Set(threats.map(t=>t.id));
    const fresh = threats.filter(t=>!lastIdsRef.current.has(t.id));
    if(fresh.length){
      setEvents(prev => [
        ...fresh.map(f=>({...f})).reverse(),
        ...prev
      ].slice(0, 80));
    }
    lastIdsRef.current = curr;
  },[threats, liveConnected]);

  // Audio beams — sim only
  useEffect(()=>{
    if(paused || liveConnected) return;
    const id = setInterval(()=>{
      setBeams(prev => {
        const target = {front:rand(0.05,0.25), right:rand(0.05,0.25), back:rand(0.04,0.18), left:rand(0.05,0.25)};
        // amplify the cone closest to nearest threat
        if(threats.length){
          const t0 = [...threats].sort((a,b)=>a.distance-b.distance)[0];
          const dir = compass(t0.angle);
          const key = dir==="FRONT"||dir==="NE"||dir==="NW" ? "front"
                   : dir==="RIGHT"||dir==="SE" ? "right"
                   : dir==="BACK"||dir==="SW" ? "back" : "left";
          target[key] = Math.min(1, target[key] + (t0.urgency==="critical"?0.75:t0.urgency==="high"?0.55:0.3));
        }
        return {
          front: prev.front + (target.front-prev.front)*0.35,
          right: prev.right + (target.right-prev.right)*0.35,
          back:  prev.back  + (target.back -prev.back )*0.35,
          left:  prev.left  + (target.left -prev.left )*0.35,
        };
      });
    }, 90);
    return ()=>clearInterval(id);
  },[paused, threats]);

  // IMU heading — slow drift
  useEffect(()=>{
    if(paused) return;
    let h = heading; let v = 0;
    const id = setInterval(()=>{
      v += (Math.random()*2-1)*0.5;
      v = Math.max(-3, Math.min(3, v));
      h = (h + v + 360) % 360;
      setHeading(h);
    }, 240);
    return ()=>clearInterval(id);
  },[paused]); // eslint-disable-line

  // System stats — drift
  useEffect(()=>{
    const id = setInterval(()=>{
      setStat(s=>({
        gemma:    Math.max(1.4, Math.min(6.5, s.gemma + (Math.random()*2-1)*0.18)),
        pipeline: Math.max(4.0, Math.min(11, s.pipeline + (Math.random()*2-1)*0.4)),
        yolo:     Math.max(48, Math.min(220, Math.round(s.yolo + (Math.random()*2-1)*8))),
        temp:     Math.max(38, Math.min(64, Math.round(s.temp + (Math.random()*2-1)*0.4))),
        batt:     Math.max(0, s.batt - (Math.random()<0.05 ? 1 : 0)),
      }));
      if(Math.random()<0.04){ setSignal(s=>Math.max(1,Math.min(4, s + (Math.random()<0.5?-1:1)))); }
    }, 600);
    return ()=>clearInterval(id);
  },[]);

  // Keyboard
  useEffect(()=>{
    function onKey(e){
      if(e.target.tagName==="INPUT"||e.target.tagName==="TEXTAREA") return;
      if(e.key===" "){ e.preventDefault(); setPaused(p=>!p); }
      else if(e.key==="m"||e.key==="M"){ setMuted(m=>!m); }
      else if(e.key==="s"||e.key==="S"){ setShowSb(s=>!s); }
      else if(e.key==="r"||e.key==="R"){ setThreats([]); setEvents([]); }
      else if(e.key==="t"||e.key==="T"){ setTweak("theme", t.theme==="light"?"dark":"light"); }
      else if(e.key==="c"||e.key==="C"){ setTweak("liveCam", !t.liveCam); }
      else if(e.key==="?"){ setShowHelp(s=>!s); }
      else if(e.key==="Escape"){ setShowHelp(false); setModalThreat(null); }
    }
    window.addEventListener("keydown", onKey);
    return ()=>window.removeEventListener("keydown", onKey);
  },[t.theme, t.liveCam]); // eslint-disable-line

  const fmtTime = (d)=>{
    const p = n => String(n).padStart(2,"0");
    return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  };
  const ms = String(now.getMilliseconds()).padStart(3,"0").slice(0,2);

  const critCount = threats.filter(th=>th.urgency==="critical").length;
  const connectionOk = signal>=2;

  return (
    <div className={"app "+(t.scanlines?"scanlines":"")}>
      {/* Top bar */}
      <div className="topbar">
        <div className="brand">
          <img src="assets/wavescape-logo.png" alt="WaveScape"/>
          <span className="ver">v1.0.4 · BUILD 2026.05</span>
        </div>
        <div className="sep"/>
        <div className="tb-item hide-sm">
          <span className={"dot "+(liveConnected?"ok":"err")}/>
          <span className="lbl">LINK</span>
          <span className="val">{liveConnected?"LIVE":"SIM"}</span>
        </div>
        <div className="sep hide-sm"/>
        <div className="tb-item hide-sm">
          <span className={"dot "+(critCount?"err":"ok")}/>
          <span className="lbl">STATE</span>
          <span className="val">{critCount?"ALERT":"NOMINAL"}</span>
        </div>
        <div className="sep hide-sm"/>
        <div className="tb-item hide-sm">
          <span className="lbl">WEARER</span>
          <span className="val">M. RIVERA / EID-21A</span>
        </div>
        <div className="tb-grow"/>
        <div className="tb-item">
          <span className="lbl">UTC-7</span>
          <span className="tb-clock">{fmtTime(now)}<span className="ms">.{ms}</span></span>
        </div>
        <div className="sep"/>
        <button
          className="theme-toggle"
          onClick={()=>setTweak("theme", t.theme==="light"?"dark":"light")}
          title={t.theme==="light"?"Switch to dark mode":"Switch to light mode"}>
          <span className="glyph"/>
          <span>{t.theme==="light"?"DAY":"NIGHT"}</span>
        </button>
        <div className="sep"/>
        <button
          onClick={()=>setShowHelp(s=>!s)}
          style={{background:"transparent",border:".5px solid var(--div-hi)",color:"var(--tx-2)",padding:"4px 10px",fontSize:9.5,fontFamily:"var(--mono)",letterSpacing:".2em",cursor:"pointer",borderRadius:2,height:26}}>
          ? HELP
        </button>
      </div>

      {/* Main grid */}
      <div className="grid">
        <HapticLog
          events={events}
          expanded={expandedEv}
          setExpanded={setExpandedEv}
          onAck={()=>{ setEvents([]); }}
        />
        <RadarPanel
          threats={threats}
          beams={beams}
          heading={heading}
          paused={paused}
          sweepEnabled={t.sweep}
          onThreatClick={setModalThreat}
        />
        <div className="cell area-right">
          <CameraFeed
            threats={threats}
            yoloDets={yoloDets}
            fps={Math.round(stat.pipeline*3)}
            yoloMs={stat.yolo}
            temp={stat.temp}
            heading={heading}
            liveCam={t.liveCam}
            onLiveCamToggle={v=>setTweak("liveCam", v)}
            liveConnected={liveConnected}
          />
          <AudioBeams beams={beams} mode={t.beamMode}/>
        </div>
        <InfoRow threats={threats} heading={heading}/>
      </div>

      {/* Status bar */}
      {showSb && <StatusBar stat={stat} beams={beams} signal={signal} muted={muted}/>}

      {/* Help overlay */}
      {showHelp && (
        <div className="help">
          <h4>KEYBOARD</h4>
          <div className="kb"><span>Pause sweep</span><kbd>Space</kbd></div>
          <div className="kb"><span>Mute haptics</span><kbd>M</kbd></div>
          <div className="kb"><span>Toggle status</span><kbd>S</kbd></div>
          <div className="kb"><span>Reset alerts</span><kbd>R</kbd></div>
          <div className="kb"><span>Toggle theme</span><kbd>T</kbd></div>
          <div className="kb"><span>Operator cam</span><kbd>C</kbd></div>
          <div className="kb"><span>Toggle help</span><kbd>?</kbd></div>
          <div className="kb"><span>Close modal</span><kbd>Esc</kbd></div>
          <h4 style={{marginTop:12}}>LEGEND</h4>
          <div className="kb"><span style={{color:"#E24B4A"}}>● Critical</span><span>&lt;2m</span></div>
          <div className="kb"><span style={{color:"#BA7517"}}>● High</span><span>2–4m</span></div>
          <div className="kb"><span style={{color:"#FFD54F"}}>● Medium</span><span>4–8m</span></div>
          <div className="kb"><span style={{color:"#888"}}>● Low</span><span>&gt;8m</span></div>
        </div>
      )}

      {/* Threat modal */}
      <ThreatModal
        threat={modalThreat}
        onClose={()=>setModalThreat(null)}
        onAck={(th)=>{ setThreats(p=>p.filter(x=>x.id!==th.id)); }}
        onTrack={(th)=>{}}
      />

      {/* Tweaks panel */}
      <TweaksPanel title="Tweaks">
        <TweakSection label="Display"/>
        <TweakRadio label="Theme" value={t.theme}
                    options={["dark","light"]}
                    onChange={v=>setTweak("theme", v)}/>
        <TweakToggle label="Operator cam" value={!!t.liveCam}
                     onChange={v=>setTweak("liveCam", v)}/>
        <TweakRadio label="Audio viz" value={t.beamMode}
                    options={["bars","polar"]}
                    onChange={v=>setTweak("beamMode", v)}/>
        <TweakToggle label="Radar sweep" value={t.sweep}
                     onChange={v=>setTweak("sweep", v)}/>
        <TweakToggle label="Scan-lines" value={t.scanlines}
                     onChange={v=>setTweak("scanlines", v)}/>
        <TweakSection label="Simulation"/>
        <TweakRadio label="Spawn rate" value={t.spawnRate}
                    options={["quiet","normal","storm"]}
                    onChange={v=>setTweak("spawnRate", v)}/>
        <TweakSection label="Accent"/>
        <TweakColor label="HUD color" value={ACCENTS[t.accent]?.hex || ACCENTS.teal.hex}
                    options={["teal","amber","cyan","rose"].map(k=>ACCENTS[k].hex)}
                    onChange={hex=>{
                      const name = Object.entries(ACCENTS).find(([_,a])=>a.hex===hex)?.[0] || "teal";
                      setTweak("accent", name);
                    }}/>
        <TweakSection label="Quick actions"/>
        <TweakButton label="Spawn critical" onClick={()=>setThreats(p=>[...p, newThreat("critical")])}/>
        <TweakButton label="Clear all" onClick={()=>{setThreats([]); setEvents([]);}}/>
        <TweakToggle label="Pause" value={paused} onChange={setPaused}/>
      </TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
