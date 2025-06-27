import React from "react"
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faUser } from '@fortawesome/free-solid-svg-icons'


const Head = () => {
  return (
    <>
      <section className='head'>
        <div className='container flexSB'>
          <div className='headlogo'>
            <h1>Cognifit</h1>
            <span>AI-Powered Mental Wellness Platform</span>
          </div>

          <div className='social'>
            <i className='headicon'><a href="/"><FontAwesomeIcon icon={faUser} /></a></i>
          </div>
        </div>
      </section>
    </>
  )
}

export default Head
