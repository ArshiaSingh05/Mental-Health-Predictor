import React from "react"
import Heading from "../../common/heading/Heading"
import "./hero.css"

const Hero = () => {
  return (
    <>
      <section className='hero'>
        <div className='container'>
          <div className='row'>
            <Heading subtitle='WELCOME TO Cognifit-AI' title='Your AI-powered mental wellness companion' />
            <p>Your mental well-being is our priority.</p>
          </div>
        </div>
      </section>

      {/* Buttons Moved Outside Hero */}
      <div className="hero-buttons">
        <a href='/' className='primary-btn'>
          GET STARTED NOW <i className='fa fa-long-arrow-alt-right'></i>
        </a>
        <a href='/pricing' className='secondary-btn'>
          VIEW PRICING <i className='fa fa-long-arrow-alt-right'></i>
        </a>
      </div>

      <div className='margin'></div>
    </>
  )
}

export default Hero
